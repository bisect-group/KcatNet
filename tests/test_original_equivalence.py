"""FastKcatNet must reproduce the authors' KcatNet (models/model_kcat.py).

Run from baselines/KcatNet:
    python -m pytest tests/test_original_equivalence.py -v
The GPU test runs the Triton kernels. Pick the GPU with CUDA_VISIBLE_DEVICES; it skips without CUDA.

Batches go through emulator_bench.dataset (edges sorted by target node) and a PyG DataLoader, like
the trainer. Graphs are real cached proteins and ligands when the local cache exists, else synthetic
graphs with the same structure (symmetric contacts, self-loops, weights in (0, 1]).

The Triton kernels need CUDA. On CPU the three kernels (edge_hidden, graph_norm, segment_stats) are
replaced by plain torch versions of their documented semantics. Everything else in FastKcatNet
(CSR reductions, block-diagonal towers, edge-list mincut loss, GCN propagation) runs unchanged.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader
from torch_scatter import segment_csr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emulator_bench import fast_models  # noqa: E402
from emulator_bench.dataset import CachedKcatDataset, compute_pna_degrees  # noqa: E402
import models.model_kcat as original_module  # noqa: E402
from models.model_kcat import KcatNet  # noqa: E402

CACHE = ROOT.parents[1] / "outputs/baselines/KcatNet/embeddings"
MODEL_KWARGS = dict(
    mol_in_channels=43, prot_in_channels=1280, prot_evo_channels=1024, hidden_channels=200,
    pre_layers=2, post_layers=1, aggregators=["mean", "min", "max", "std"],
    scalers=["identity", "amplification", "linear"], total_layer=3, K=[3, 10, 30], heads=5,
    dropout=0, dropout_attn_score=0.2,
)  # config_KcatNet.json
REGRESSION_WEIGHT, CLUSTER_WEIGHT = 0.9, 0.1  # emulator_bench/train_single_target_tvt.py


class DictStore:
    def __init__(self, items):
        self.items = items

    def get(self, key):
        return self.items[str(key)]


def synthetic_protein(rng, n):
    pairs = {(i, i + d) for i in range(n) for d in (1, 2) if i + d < n}
    pairs |= {tuple(sorted(rng.choice(n, 2, replace=False))) for _ in range(2 * n)}
    pairs = np.array(sorted(pairs)).T
    weight = rng.uniform(0.05, 1.0, pairs.shape[1]).astype(np.float32)
    edge_index = np.concatenate([pairs, pairs[::-1], np.tile(np.arange(n), (2, 1))], axis=1)
    edge_weight = np.concatenate([weight, weight, np.ones(n, np.float32)])
    order = np.lexsort((edge_index[1], edge_index[0]))  # row-major, as the cache stores it
    return {
        "esm": rng.normal(size=(n, 1280)).astype(np.float16),
        "prot5": rng.normal(size=(n, 1024)).astype(np.float32),
        "edge_index": edge_index[:, order].astype(np.int32),
        "edge_weight": edge_weight[order],
    }


def synthetic_ligand(rng, n):
    return {
        "atom_idx": rng.integers(0, 20, n).astype(np.int16),
        "atom_feature": rng.integers(0, 2, (n, 43)).astype(np.float32),
        "total_fea": rng.normal(size=1024).astype(np.float32),
    }


def cached_items(count=3, max_res=80):
    proteins, ligands = [], []
    for path in sorted((CACHE / "proteins_v3").glob("*/*.npz")):
        with np.load(path) as data:
            item = {key: data[key] for key in data.files}
        if len(item["esm"]) <= max_res:
            proteins.append(item)
        if len(proteins) == count:
            break
    for path in sorted((CACHE / "ligands_v3").glob("*/*.npz"))[:count]:
        with np.load(path) as data:
            ligands.append({key: data[key] for key in data.files})
    if len(proteins) < count or len(ligands) < count:
        pytest.skip("no cached embeddings")
    return proteins, ligands


@pytest.fixture(params=["synthetic", "cached"])
def dataset(request):
    if request.param == "synthetic":
        rng = np.random.default_rng(0)
        proteins = [synthetic_protein(rng, n) for n in (23, 31, 14)]
        ligands = [synthetic_ligand(rng, n) for n in (9, 12, 6)]
    else:
        proteins, ligands = cached_items()
    seqs = ["A" * (len(p["esm"]) - 1) + "CDE"[i] for i, p in enumerate(proteins)]
    smiles = [f"mol{i}" for i in range(len(ligands))]
    # The fourth row repeats protein 0 to exercise the degree-histogram counts.
    frame = pd.DataFrame(
        {
            "sequence": seqs + seqs[:1],
            "smiles": smiles + smiles[:1],
            "log10_value": [0.3, -1.2, 2.0, 0.7],
        }
    )
    return CachedKcatDataset(
        frame, DictStore(dict(zip(seqs, proteins))), DictStore(dict(zip(smiles, ligands)))
    )


def make_batch(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), follow_batch=["mol_x", "prot_node_esm"])
    return next(iter(loader))


def to_device(batch, device, dtype):
    batch = batch.to(device)
    for key in ("prot_node_esm", "prot_node_prot5", "mol_x_feat", "mol_total_fea", "prot_edge_weight", "reg_y"):
        batch[key] = batch[key].to(dtype)
    return batch


def inputs(batch):
    return dict(
        mol_x=batch.mol_x,
        mol_x_feat=batch.mol_x_feat,
        mol_total_fea=batch.mol_total_fea,
        residue_esm=batch.prot_node_esm,
        residue_prot5=batch.prot_node_prot5,
        residue_edge_index=batch.prot_edge_index,
        residue_edge_weight=batch.prot_edge_weight,
        mol_batch=batch.mol_x_batch,
        prot_batch=batch.prot_node_esm_batch,
    )


def run_original(model, batch):
    return model(**inputs(batch))


def run_fast(model, batch):
    cpu_sizes = batch.prot_num_nodes.cpu()
    return model(**inputs(batch), prot_num_nodes=batch.prot_num_nodes, max_res=int(cpu_sizes.max()))


def objective(model, run, batch):
    """Trainer loss: weighted MSE plus weighted cluster loss. Returns (pred, cluster_loss, loss, grads)."""
    model.zero_grad()
    pred, cluster_loss = run(model, batch)
    mse = torch.nn.functional.mse_loss(pred.squeeze(-1), batch.reg_y)
    loss = REGRESSION_WEIGHT * mse + CLUSTER_WEIGHT * cluster_loss
    loss.backward()
    grads = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
    return pred.detach(), cluster_loss.detach(), loss.detach(), grads


def build_models(dataset, device, dtype):
    prot_deg = compute_pna_degrees(dataset)
    torch.manual_seed(0)
    original = KcatNet(prot_deg.to(device), device=device, **MODEL_KWARGS).to(device)
    torch.manual_seed(0)
    fast = fast_models.FastKcatNet(prot_deg.to(device), device=device, **MODEL_KWARGS).to(device)
    # Same constructor seed gives the same initial weights.
    for (name, a), (_, b) in zip(original.state_dict().items(), fast.state_dict().items()):
        assert torch.equal(a, b), name
    # Perturbed weights so no layer is trivially the identity.
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in original.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=generator).to(device) * p.abs().mean())
    # Parameter names and shapes must match one to one: strict loading fails on any unmapped name.
    fast.load_state_dict(original.state_dict(), strict=True)
    assert list(original.state_dict()) == list(fast.state_dict())
    original.to(dtype).eval()
    fast.to(dtype).eval()
    return original, fast


def assert_close(actual, expected, rtol, what, floor=0.0):
    """Max abs error relative to max |expected|. `floor` is the smallest scale that counts as non-zero."""
    scale = max(expected.abs().max().item(), floor, 1e-30)
    error = (actual - expected).abs().max().item() / scale
    assert error <= rtol, f"{what}: max error {error:.3e} of max |value| exceeds {rtol:.1e}"


def check_grads(actual, expected, rtol):
    assert actual.keys() == expected.keys()
    # The last bias of each attention score MLP only shifts a softmax input, so its true gradient is 0
    # and the computed value is rounding noise. The floor keeps such entries from failing on noise.
    floor = 1e-2 * max(g.abs().max().item() for g in expected.values() if g is not None)
    for name in expected:
        assert (actual[name] is None) == (expected[name] is None), name
        if expected[name] is not None:
            assert_close(actual[name], expected[name], rtol, f"grad {name}", floor * rtol)


def compare(original, fast, batch, rtol_out, rtol_grad):
    ref = objective(original, run_original, batch)
    out = objective(fast, run_fast, batch)
    for what, a, b in zip(("prediction", "cluster loss", "total loss"), out[:3], ref[:3]):
        assert_close(a, b, rtol_out, what)
    check_grads(out[3], ref[3], rtol_grad)
    assert sum(g is not None for g in ref[3].values()) > 50  # most of the model is exercised


def torch_edge_hidden(node_i, node_j, edge, src, ptr):
    dst = torch.repeat_interleave(torch.arange(ptr.numel() - 1, device=ptr.device), ptr[1:] - ptr[:-1])
    return torch.relu(node_i[dst] + node_j[src] + edge)


def torch_segment_stats(x, ptr):
    return (
        segment_csr(x, ptr, reduce="sum"),
        segment_csr(x * x, ptr, reduce="sum"),
        segment_csr(x, ptr, reduce="min"),
        segment_csr(x, ptr, reduce="max"),
    )


def torch_graph_norm(gn, x, ptr):
    batch = torch.repeat_interleave(torch.arange(ptr.numel() - 1, device=ptr.device), ptr[1:] - ptr[:-1])
    return gn(x, batch)


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(fast_models, "edge_hidden", torch_edge_hidden)
    monkeypatch.setattr(fast_models, "segment_stats", torch_segment_stats)
    monkeypatch.setattr(fast_models, "graph_norm", torch_graph_norm)


@pytest.fixture
def float64_adjacency(monkeypatch):
    """The original builds its dense adjacency in float32; cast it so the original can run in float64.

    The CPU pyg_lib softmax_csr rejects float64, so the fast model's softmax over `ptr` uses the batch vector instead.
    """
    softmax = fast_models.softmax

    def softmax_by_index(src, index=None, ptr=None, **kwargs):
        if ptr is not None:
            index = torch.repeat_interleave(torch.arange(ptr.numel() - 1), ptr[1:] - ptr[:-1])
        return softmax(src, index, **kwargs)

    monkeypatch.setattr(fast_models, "softmax", softmax_by_index)
    to_dense_adj = original_module.to_dense_adj
    monkeypatch.setattr(original_module, "to_dense_adj", lambda *a, **k: to_dense_adj(*a, **k).double())
    return None


# The cluster loss accumulates in a float32 scalar in both models, so the fp64 loss terms agree to
# float32 precision only.
@pytest.mark.parametrize(
    "dtype, rtol_out, rtol_grad",
    [(torch.float64, 1e-6, 1e-7), (torch.float32, 1e-5, 2e-3)],
)
def test_cpu_matches_original(dataset, cpu_kernels, request, dtype, rtol_out, rtol_grad):
    if dtype == torch.float64:
        request.getfixturevalue("float64_adjacency")
    original, fast = build_models(dataset, "cpu", dtype)
    batch = to_device(make_batch(dataset), "cpu", dtype)
    compare(original, fast, batch, rtol_out, rtol_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the Triton kernels")
def test_gpu_triton_matches_original(dataset):
    device = "cuda:0"
    original, fast = build_models(dataset, device, torch.float32)
    batch = to_device(make_batch(dataset), device, torch.float32)
    compare(original, fast, batch, rtol_out=1e-4, rtol_grad=1e-3)


def test_dataset_edge_order_keeps_original_output(dataset, float64_adjacency):
    batch = make_batch(dataset)
    dst = batch.prot_edge_index[1]
    assert (dst[1:] >= dst[:-1]).all(), "edges are not sorted by target node"
    # Same edge set as the cache, only reordered.
    raw = np.concatenate(
        [dataset.protein_store.get(s)["edge_index"].astype(np.int64) for s in dataset.frame.sequence], axis=1
    )
    sizes = batch.prot_num_nodes.tolist()
    offsets = np.repeat(np.cumsum([0] + sizes[:-1]), [dataset.protein_store.get(s)["edge_index"].shape[1] for s in dataset.frame.sequence])
    raw = raw + offsets
    key = lambda e: np.lexsort((e[1], e[0]))
    sorted_edges = batch.prot_edge_index.numpy()
    assert np.array_equal(raw[:, key(raw)], sorted_edges[:, key(sorted_edges)])

    original, _ = build_models(dataset, "cpu", torch.float64)
    batch = to_device(batch, "cpu", torch.float64)
    perm = torch.randperm(dst.numel(), generator=torch.Generator().manual_seed(2))
    shuffled = batch.clone()
    shuffled.prot_edge_index = batch.prot_edge_index[:, perm]
    shuffled.prot_edge_weight = batch.prot_edge_weight[perm]
    ref = objective(original, run_original, batch)
    other = objective(original, run_original, shuffled)
    for what, a, b in zip(("prediction", "cluster loss", "total loss"), other[:3], ref[:3]):
        assert_close(a, b, 1e-6, what)
    check_grads(other[3], ref[3], 1e-6)


def test_pna_degree_histogram(dataset):
    batch = make_batch(dataset)
    in_degree = torch.bincount(batch.prot_edge_index[1], minlength=batch.prot_node_esm.size(0))
    expected = torch.bincount(in_degree)
    assert torch.equal(compute_pna_degrees(dataset), expected)
