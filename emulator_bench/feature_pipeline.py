import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import esm.pretrained as esm_pretrained
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from rdkit import Chem
from rdkit import RDLogger
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops, to_undirected
from transformers import T5EncoderModel, T5Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
UTILS_ROOT = REPO_ROOT / "utils"
if str(UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(UTILS_ROOT))

from build_vocab import WordVocab


VOCAB_PATH = REPO_ROOT / "utils" / "vocab.pkl"
SMILES_TRFM_PATH = REPO_ROOT / "utils" / "trfm_12_23000.pkl"
RDKIT_FEATURE_DEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(RDKIT_FEATURE_DEF)
RDLogger.DisableLog("rdApp.warning")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size,
        )
        self.out = nn.Linear(hidden_size, out_size)

    def encode(self, src):
        """Encode one molecule: src is (seq_len, 1). The positional encoding adds pe[:, :1] (original quirk)."""
        embedded = self.pe(self.embed(src))
        output = embedded
        for idx in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[idx](output, None)
        penultimate = output.detach().cpu().numpy()
        output = self.trfm.encoder.layers[-1](output, None)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)
        output = output.detach().cpu().numpy()
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penultimate[0, :, :]])


def load_prot_t5(model_name_or_path: str, device: torch.device) -> Tuple[T5EncoderModel, T5Tokenizer]:
    model = T5EncoderModel.from_pretrained(model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
    model = model.to(device=device, dtype=torch.float16)
    return model.eval(), tokenizer


def load_esm_model(device: torch.device):
    model, alphabet = esm_pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()
    if device.type == "cuda":
        model = model.to(device)
    return model, alphabet.get_batch_converter()


def load_smiles_transformer(device: torch.device | None = None):
    device = device or torch.device("cpu")
    model = TrfmSeq2seq(45, 256, 45, 4)
    state_dict = torch.load(SMILES_TRFM_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_prot_t5_batches(
    sequences: Sequence[str],
    max_residues: int = 4000,
    max_batch: int = 100,
) -> List[List[str]]:
    """Same batching rule as the original `get_embeddings`: longest first, flush when full."""
    ordered = sorted(sequences, key=len, reverse=True)
    batches: List[List[str]] = []
    batch: List[str] = []
    for idx, seq in enumerate(ordered, 1):
        batch.append(seq)
        residues = sum(len(item) for item in batch) + len(seq)
        if len(batch) >= max_batch or residues >= max_residues or idx == len(ordered):
            batches.append(batch)
            batch = []
    return batches


def embed_prot_t5_batch(
    model: T5EncoderModel,
    tokenizer: T5Tokenizer,
    sequences: Sequence[str],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    tokenized = [" ".join(list(seq)) for seq in sequences]
    encoding = tokenizer(tokenized, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(encoding["input_ids"], device=device)
    attention_mask = torch.tensor(encoding["attention_mask"], device=device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask).last_hidden_state

    embedded = {}
    for idx, sequence in enumerate(sequences):
        residue_count = len(sequence)
        array = outputs[idx, :residue_count].detach().cpu().numpy()
        embedded[sequence] = array
    return embedded


def _contact_map(contact_probs: torch.Tensor, threshold: float = 0.5):
    num_residues = int(contact_probs.shape[0])
    adjacency = (contact_probs >= threshold).long()
    edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index
    edge_weight = contact_probs[row, col].float()

    seq_edge_head_1 = torch.stack([torch.arange(num_residues)[:-1], (torch.arange(num_residues) + 1)[:-1]])
    seq_edge_tail_1 = torch.stack([torch.arange(num_residues)[1:], (torch.arange(num_residues) - 1)[1:]])
    seq_weight_1 = torch.full((seq_edge_head_1.size(1) + seq_edge_tail_1.size(1),), threshold, dtype=torch.float32)

    seq_edge_head_2 = torch.stack([torch.arange(num_residues)[:-2], (torch.arange(num_residues) + 2)[:-2]])
    seq_edge_tail_2 = torch.stack([torch.arange(num_residues)[2:], (torch.arange(num_residues) - 2)[2:]])
    seq_weight_2 = torch.full((seq_edge_head_2.size(1) + seq_edge_tail_2.size(1),), threshold, dtype=torch.float32)

    edge_index = torch.cat([edge_index, seq_edge_head_1, seq_edge_tail_1, seq_edge_head_2, seq_edge_tail_2], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_weight_1, seq_weight_2], dim=-1)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce="max")
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="max")
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0)
    return edge_index, edge_weight


def esm_extract_last_layer(
    model,
    batch_converter,
    sequence: str,
    device: torch.device,
    layer: int = 33,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(sequence) <= 700:
        _, _, tokens = batch_converter([("protein", sequence)])
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[layer], return_contacts=True)
        token_repr = results["representations"][layer][0, 1 : len(sequence) + 1].detach().cpu()
        contact_probs = results["contacts"][0].detach().cpu()
        return token_repr, contact_probs

    contact_probs = np.zeros((len(sequence), len(sequence)), dtype=np.float32)
    token_repr = np.zeros((len(sequence), 1280), dtype=np.float32)
    interval = 350
    n_windows = math.ceil(len(sequence) / interval)

    for window_idx in range(n_windows):
        start = window_idx * interval
        end = min((window_idx + 2) * interval, len(sequence))
        sub_sequence = sequence[start:end]

        _, _, tokens = batch_converter([("protein", sub_sequence)])
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[layer], return_contacts=True)

        sub_contacts = results["contacts"][0].detach().cpu().numpy()
        sub_repr = results["representations"][layer][0, 1 : len(sub_sequence) + 1].detach().cpu().numpy()

        row, col = np.where(contact_probs[start:end, start:end] != 0)
        row = row + start
        col = col + start
        contact_probs[start:end, start:end] += sub_contacts
        if len(row) > 0:
            contact_probs[row, col] = contact_probs[row, col] / 2.0

        existing = np.where(token_repr[start:end].sum(axis=-1) != 0)[0] + start
        token_repr[start:end] += sub_repr
        if len(existing) > 0:
            token_repr[existing] = token_repr[existing] / 2.0

        if end == len(sequence):
            break

    return torch.from_numpy(token_repr), torch.from_numpy(contact_probs)


def protein_cache_item(
    sequence: str,
    prot5_array: np.ndarray,
    esm_model,
    batch_converter,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    esm_repr, contact_probs = esm_extract_last_layer(esm_model, batch_converter, sequence, device=device)
    edge_index, edge_weight = _contact_map(contact_probs.float())

    return {
        "prot5": prot5_array.astype(np.float32, copy=False),
        "esm": esm_repr.numpy().astype(np.float16, copy=False),
        "edge_index": edge_index.numpy().astype(np.int32, copy=False),
        "edge_weight": edge_weight.numpy().astype(np.float32, copy=False),
    }


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda value: x == value, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda value: x == value, allowable_set))


def atom_features(atom) -> np.ndarray:
    encoding = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(
        atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    encoding += one_of_k_encoding_unk(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "other",
        ],
    )
    encoding += [atom.GetIsAromatic()]
    try:
        encoding += one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
    except Exception:
        encoding += [0, 0] + [atom.HasProp("_ChiralityPossible")]
    return np.asarray(encoding, dtype=np.float32)


ATOM_CODES = {
    5: 0,
    6: 1,
    7: 2,
    8: 3,
    15: 4,
    16: 5,
    34: 6,
}
for atomic_num in [9, 17, 35, 53]:
    ATOM_CODES[atomic_num] = 7
for atomic_num in ([3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + list(range(55, 84)) + list(range(87, 104))):
    ATOM_CODES[atomic_num] = 8


def _atom_idx(atom) -> int:
    return int(ATOM_CODES.get(atom.GetAtomicNum(), -1) + 1)


def ligand_graph_features(smiles: str) -> Dict[str, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atom_idx = []
    atom_feat = []
    for atom in mol.GetAtoms():
        atom_idx.append(_atom_idx(atom))
        atom_feat.append(atom_features(atom))
    return {
        "atom_idx": np.asarray(atom_idx, dtype=np.int16),
        "atom_feature": np.asarray(atom_feat, dtype=np.float32),
    }


def split_smiles(smiles: str) -> str:
    arr = []
    i = 0
    two_char_tokens = {
        "Cl", "Ca", "Cu", "Br", "Be", "Ba", "Bi", "Si", "Se", "Sr", "Na", "Ni", "Rb", "Ra",
        "Xe", "Li", "Al", "As", "Ag", "Au", "Mg", "Mn", "Te", "Zn", "si", "se", "te", "He",
        "+2", "+3", "+4", "-2", "-3", "-4", "Kr", "Fe",
    }
    while i < len(smiles) - 1:
        if smiles[i] == "%":
            arr.append(smiles[i : i + 3])
            i += 3
            continue
        token = smiles[i : i + 2]
        if token in two_char_tokens:
            arr.append(token)
            i += 2
        else:
            arr.append(smiles[i])
            i += 1
    if i == len(smiles) - 1:
        arr.append(smiles[i])
    return " ".join(arr)


@lru_cache(maxsize=1)
def _smiles_vocab():
    return WordVocab.load_vocab(VOCAB_PATH)


def smiles_to_vec(smiles: str, trfm_model: TrfmSeq2seq) -> np.ndarray:
    pad_index, unk_index, eos_index, sos_index = 0, 1, 2, 3
    vocab = _smiles_vocab()
    device = next(trfm_model.parameters()).device

    tokens = split_smiles(smiles).split()
    if len(tokens) > 218:
        tokens = tokens[:109] + tokens[-109:]
    ids = [sos_index] + [vocab.stoi.get(token, unk_index) for token in tokens] + [eos_index]
    ids += [pad_index] * (220 - len(ids))
    encoded = trfm_model.encode(torch.tensor([ids], dtype=torch.long, device=device).t())
    return np.asarray(encoded[0], dtype=np.float32)


def ligand_cache_item(smiles: str, trfm_model: TrfmSeq2seq) -> Dict[str, np.ndarray]:
    item = ligand_graph_features(smiles)
    item["total_fea"] = smiles_to_vec(smiles, trfm_model)
    return item
