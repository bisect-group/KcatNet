import hashlib
import os
import sys
import zipfile
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
UTILS_ROOT = REPO_ROOT / "utils"
if str(UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(UTILS_ROOT))

from emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_path
from Kcat_Dataset import MultiGraphData


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, zlib.error) as exc:
        raise RuntimeError(
            f"Corrupted cached embedding file: {path}. Rebuild the cache for this split with "
            "`cache_embeddings.py --overwrite` or delete this file and rerun caching."
        ) from exc


def _pna_cache_key(train_path: Path) -> str:
    resolved = train_path.resolve()
    stat = resolved.stat()
    payload = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pna_degree_cache_path(cache_root: Path, train_path: Path) -> Path:
    key = _pna_cache_key(train_path)
    return Path(cache_root) / key[:2] / f"{key}.pt"


class ProteinEmbeddingStore:
    def __init__(self, embeddings_dir: Path, sequences=None, preload: bool = False, max_items: int = 256, max_len: int = 1000):
        self.embeddings_dir = Path(embeddings_dir)
        self.max_items = max(1, int(max_items))
        self.max_len = max_len
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        if preload and sequences is not None:
            unique_sequences = sorted({normalize_sequence(sequence, max_len=self.max_len) for sequence in sequences})
            iterator = progress(unique_sequences, desc="Preloading protein embeddings", unit="protein")
            for normalized in iterator:
                path = protein_cache_path(self.embeddings_dir, normalized, max_len=self.max_len)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached protein embedding: {path}")
                self._cache[normalized] = _load_npz(path)

    def get(self, sequence: str) -> Dict[str, np.ndarray]:
        normalized = normalize_sequence(sequence, max_len=self.max_len)
        if normalized in self._cache:
            self._cache.move_to_end(normalized)
            return self._cache[normalized]

        path = protein_cache_path(self.embeddings_dir, sequence, max_len=self.max_len)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached protein embedding: {path}")

        item = _load_npz(path)
        self._cache[normalized] = item
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item


class LigandEmbeddingStore:
    def __init__(self, embeddings_dir: Path, smiles_values=None, preload: bool = True):
        self.embeddings_dir = Path(embeddings_dir)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        if preload and smiles_values is not None:
            unique_smiles = sorted({str(smiles) for smiles in smiles_values})
            for smiles in unique_smiles:
                path = ligand_cache_path(self.embeddings_dir, smiles)
                if not path.exists():
                    raise FileNotFoundError(f"Missing cached ligand embedding: {path}")
                self._cache[smiles] = _load_npz(path)

    def get(self, smiles: str) -> Dict[str, np.ndarray]:
        smiles = str(smiles)
        if smiles in self._cache:
            return self._cache[smiles]

        path = ligand_cache_path(self.embeddings_dir, smiles)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached ligand embedding: {path}")
        item = _load_npz(path)
        self._cache[smiles] = item
        return item


class CachedKcatDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        protein_store: ProteinEmbeddingStore,
        ligand_store: LigandEmbeddingStore,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: Optional[str] = "log10_value",
        protein_max_len: int = 1000,
    ):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.ligand_store = ligand_store
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.protein_max_len = protein_max_len

    def len(self):
        return len(self.frame)

    def get(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        raw_smiles = str(row[self.smiles_col])
        raw_sequence = str(row[self.sequence_col])
        normalized_sequence = normalize_sequence(raw_sequence, max_len=self.protein_max_len)

        ligand = self.ligand_store.get(raw_smiles)
        protein = self.protein_store.get(raw_sequence)
        # The model reduces over segments, so edges are sorted by target node.
        order = np.argsort(protein["edge_index"][1], kind="stable")

        reg_y = None
        if self.target_col is not None and self.target_col in self.frame.columns:
            value = row[self.target_col]
            if pd.notna(value):
                reg_y = torch.tensor(float(value), dtype=torch.float32)

        return MultiGraphData(
            mol_x=torch.from_numpy(ligand["atom_idx"]).long().reshape(-1, 1),
            mol_x_feat=torch.from_numpy(ligand["atom_feature"]).float(),
            mol_total_fea=torch.from_numpy(ligand["total_fea"]).float().reshape(1, -1),
            prot_node_esm=torch.from_numpy(protein["esm"]),
            prot_node_prot5=torch.from_numpy(protein["prot5"]).float(),
            prot_node_pos=torch.arange(len(normalized_sequence)).reshape(-1, 1),
            prot_seq=normalized_sequence,
            prot_edge_index=torch.from_numpy(protein["edge_index"][:, order]).long(),
            prot_edge_weight=torch.from_numpy(protein["edge_weight"][order]).float(),
            prot_num_nodes=len(normalized_sequence),
            reg_y=reg_y,
            mol_key=raw_smiles,
            prot_key=normalized_sequence,
        )


def compute_pna_degrees(
    dataset: CachedKcatDataset,
    show_progress: bool = False,
    desc_prefix: str = "PNA degree cache",
) -> torch.Tensor:
    """In-degree histogram over all residues of the rows; each unique sequence is read once."""
    frame, store = dataset.frame, dataset.protein_store
    counts = frame[dataset.sequence_col].astype(str).value_counts()
    sequences = counts.index
    if show_progress:
        sequences = progress(sequences, total=len(counts), desc=desc_prefix, unit="sequence")
    degree_hist = torch.zeros(1, dtype=torch.long)
    for sequence in sequences:
        edge_index = torch.from_numpy(store.get(sequence)["edge_index"]).long()
        num_nodes = len(normalize_sequence(sequence, max_len=dataset.protein_max_len))
        hist = torch.bincount(torch.bincount(edge_index[1], minlength=num_nodes)) * int(counts[sequence])
        if hist.numel() > degree_hist.numel():
            degree_hist = torch.nn.functional.pad(degree_hist, (0, hist.numel() - degree_hist.numel()))
        degree_hist[: hist.numel()] += hist
    return degree_hist


def get_or_compute_pna_degrees(
    dataset: CachedKcatDataset,
    train_path: Path,
    cache_root: Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> torch.Tensor:
    cache_path = pna_degree_cache_path(cache_root, train_path)
    if cache_path.exists() and not overwrite:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if verbose:
            print(f"Loaded cached PNA degree histogram: {cache_path}", flush=True)
        return payload["prot_deg"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Building PNA degree histogram from train split: {train_path}", flush=True)
    prot_deg = compute_pna_degrees(dataset, show_progress=verbose)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "prot_deg": prot_deg.cpu(),
            "train_path": str(Path(train_path).resolve()),
        },
        tmp_path,
    )
    os.replace(tmp_path, cache_path)
    if verbose:
        print(f"Saved cached PNA degree histogram: {cache_path}", flush=True)
    return prot_deg
