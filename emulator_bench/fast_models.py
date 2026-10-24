"""Faster KcatNet with the same parameters and outputs as models.model_kcat.KcatNet.

Protein edges arrive sorted by target node, so every neighbourhood and every graph is a contiguous
segment. All reductions are CSR segment reductions (no atomics) and each tensor shared by the three
layers (degrees, pointers, edge features) is built once per batch. Sizes come from the CPU batch, so
the forward pass never reads GPU tensors. The mincut loss uses the edge list instead of the dense
residue adjacency. The results equal the original model up to float summation order.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax
from torch_scatter import segment_csr

from models.model_kcat import KcatNet
from emulator_bench.triton_kernels import edge_hidden, graph_norm, segment_stats
from models.Mol_pool import MotifPool

STD_EPS = 1e-5


class Graph(NamedTuple):
    src: torch.Tensor
    dst: torch.Tensor
    edge_ptr: torch.Tensor  # (N + 1) start of each target node's edges
    graph_edge_ptr: torch.Tensor  # (B + 1) start of each protein's edges
    node_ptr: torch.Tensor  # (B + 1) start of each protein's residues
    batch: torch.Tensor  # (N) protein of each residue
    deg: torch.Tensor  # (N) in-degree, float
    deg_out: torch.Tensor  # (N) out-degree, float
    dis: torch.Tensor  # (N) deg ** -0.5
    log_deg: torch.Tensor  # (N) log(deg + 1)


def _ptr(index, size):
    """CSR pointer of a sorted index vector with values in [0, size)."""
    counts = torch.zeros(size, dtype=torch.long, device=index.device).scatter_add_(
        0, index, torch.ones_like(index)
    )
    return F.pad(counts.cumsum(0), (1, 0))


def _build_graph(edge_index, node_ptr, batch, num_nodes, dtype):
    src, dst = edge_index[0], edge_index[1]
    edge_ptr = _ptr(dst, num_nodes)
    deg = (edge_ptr[1:] - edge_ptr[:-1]).to(dtype)
    deg_out = torch.zeros(num_nodes, dtype=dtype, device=src.device).scatter_add_(0, src, torch.ones_like(src, dtype=dtype))
    return Graph(
        src, dst, edge_ptr, edge_ptr[node_ptr], node_ptr, batch, deg, deg_out, deg.pow(-0.5), torch.log(deg + 1)
    )


def _stack(modules, name):
    return torch.stack([getattr(m, name) for m in modules])


def _block_diag_parts(w):
    """Block-diagonal (towers * in, towers * out) matrix from stacked (towers, out, in) weights."""
    return torch.block_diag(*w.transpose(1, 2))


def _block_diag(modules):
    """Block-diagonal (towers * in, towers * out) matrix of the tower Linear layers."""
    return torch.block_diag(*[m.weight.t() for m in modules])


def _pna(conv, x, g, rbf):
    """PNAConv forward. The first message layer is split by input block (node, node, edge) so only the
    edge term runs per edge, and the tower layers run as block-diagonal matrices."""
    towers, f = conv.towers, conv.F_in
    n = x.size(0)
    pre1 = [nn[0] for nn in conv.pre_nns]
    w1 = _stack(pre1, "weight")
    bias1 = torch.cat([m.bias for m in pre1])
    node_i = F.linear(x, _block_diag_parts(w1[..., :f]).t(), bias1)
    node_j = x @ _block_diag_parts(w1[..., f : 2 * f])
    edge = conv.edge_encoder(rbf) @ torch.cat(list(w1[..., 2 * f :].transpose(1, 2)), dim=1)
    h = edge_hidden(node_i, node_j, edge, g.src, g.edge_ptr)
    pre2 = [nn[2] for nn in conv.pre_nns]
    m = torch.addmm(torch.cat([q.bias for q in pre2]), h, _block_diag(pre2))

    total, total_sq, low, high = segment_stats(m, g.edge_ptr)
    deg = g.deg.unsqueeze(-1)
    mean = total / deg
    var = total_sq / deg - mean * mean
    std = var.clamp(min=STD_EPS).sqrt()
    std = std.masked_fill(std <= math.sqrt(STD_EPS), 0.0)
    avg = conv.aggr_module.avg_deg
    amp = (g.log_deg / avg["log"]).view(1, -1, 1)
    lin = (g.deg / avg["lin"]).view(1, -1, 1)
    post = [nn[0] for nn in conv.post_nns]
    w = _stack(post, "weight")
    a = 4 * f
    # Rows of z: x, mean, min, max, std. Columns of the product: identity, amplification and linear scaler outputs.
    zero = w.new_zeros(towers, f, 2 * f)
    w_all = torch.cat(
        [
            torch.cat([w[..., :f].transpose(1, 2), zero], dim=-1),
            torch.cat([w[..., f + i * a : f + (i + 1) * a].transpose(1, 2) for i in range(3)], dim=-1),
        ],
        dim=1,
    )
    parts = (x, mean, low, high, std)
    z = torch.cat([p.view(n, towers, f).transpose(0, 1) for p in parts], dim=-1)
    c = _TowerMM.apply(z, w_all)
    y = c[..., :f] + amp * c[..., f : 2 * f] + lin * c[..., 2 * f :] + _stack(post, "bias").unsqueeze(1)
    return conv.lin(y.transpose(0, 1).reshape(n, towers * f))


class _TowerMM(torch.autograd.Function):
    """Batched matmul whose weight gradient splits the long row dimension across the batch (the stock kernel leaves most SMs idle)."""

    CHUNKS = 16

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return torch.bmm(x, w)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        towers, rows, k = x.shape
        pad = (0, 0, 0, -rows % _TowerMM.CHUNKS)
        xs = F.pad(x, pad).view(towers * _TowerMM.CHUNKS, -1, k)
        gs = F.pad(grad, pad).view(towers * _TowerMM.CHUNKS, -1, grad.size(-1))
        grad_w = torch.bmm(xs.transpose(1, 2), gs).view(towers, _TowerMM.CHUNKS, k, -1).sum(1)
        return torch.bmm(grad, w.transpose(1, 2)), grad_w


def _propagate(h, g):
    """Symmetric-normalised neighbour sum (GCN propagation with the self-loops already in the edges)."""
    h = h * g.dis.unsqueeze(-1)
    return segment_csr(h.index_select(0, g.src), g.edge_ptr, reduce="sum") * g.dis.unsqueeze(-1)


def _cluster(cluster, x, g):
    """GCNCluster forward; the wider first layer aggregates before its linear map."""
    first, last = cluster.Conv_layers
    h = F.relu(F.linear(_propagate(x, g), first.lin.weight, first.bias))
    h = _propagate(F.linear(h, last.lin.weight), g) + last.bias
    return cluster.out_ln(h)


def _rbf(D, D_max, D_count):
    D_mu = torch.linspace(0.0, D_max, D_count, device=D.device).view(1, -1)
    return torch.exp(-(((D.clamp(max=D_max).unsqueeze(-1) - D_mu) / (D_max / D_count)) ** 2))


def _dense(x, batch_size, max_res, res_idx):
    out = x.new_zeros(batch_size * max_res, x.size(-1))
    out[res_idx] = x
    return out.view(batch_size, max_res, -1)


class FastMotifPool(MotifPool):
    def forward(self, x, mol_batch, batch_size=None):
        H, C = self.heads, self.hidden_dim
        atom_feature = x.view(-1, H, C)
        score = torch.cat([mlp(a) for a, mlp in zip(atom_feature.unbind(1), self.score_proj)], dim=-1)
        alpha = softmax(score, mol_batch, num_nodes=batch_size)
        drug_feat = global_add_pool((atom_feature * alpha.view(-1, H, 1)).view(-1, H * C), mol_batch, batch_size)
        return drug_feat, alpha


class FastKcatNet(KcatNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Switching the class keeps the parameters and their initialisation unchanged.
        for pool in self.mol_pools:
            pool.__class__ = FastMotifPool

    def forward(
        self,
        mol_x,
        mol_x_feat,
        mol_total_fea,
        residue_esm,
        residue_prot5,
        residue_edge_index,
        residue_edge_weight,
        mol_batch,
        prot_batch,
        prot_num_nodes,
        max_res,
    ):
        """`prot_num_nodes` is the (B,) residue count of each protein, `max_res` its CPU maximum."""
        device = residue_esm.device
        batch_size, num_res = prot_num_nodes.size(0), residue_esm.size(0)
        node_ptr = F.pad(prot_num_nodes.cumsum(0), (1, 0))
        offset = torch.arange(batch_size, device=device) * max_res - node_ptr[:-1]
        res_idx = torch.arange(num_res, device=device) + offset[prot_batch]
        g = _build_graph(residue_edge_index, node_ptr, prot_batch, num_res, residue_edge_weight.dtype)
        mol_ptr = _ptr(mol_batch, batch_size)

        cluster_loss = torch.zeros((), device=device)

        residue_ini = torch.cat(
            [F.relu(self.seq_embed_prot5(residue_prot5)), F.relu(self.seq_embed_esm(residue_esm))], dim=-1
        )
        residue_x = F.relu(self.seq_embed(residue_ini))
        rbf = _rbf(residue_edge_weight, 1.0, self.prot_edge_dim)

        atom_x = self.atom_type_embed(mol_x.squeeze()) + F.relu(self.atom_feat_embed(mol_x_feat))
        mol_total_fea = F.relu(self.mol_embed(mol_total_fea))
        mol_total_fea = self.norm(self.mol_embed2(mol_total_fea))

        res_feas, res_feas2, mol_feas, cluster_feas = [], [], [], []

        for idx in range(self.total_layer):
            num_cluster = self.num_cluster[idx]
            residue_x = graph_norm(self.GN, residue_x, node_ptr)
            atom_x = graph_norm(self.GN, atom_x, mol_ptr)

            block = self.prot_convs[idx]
            residue_x = residue_x + F.relu(_pna(block.conv, residue_x, g, rbf))
            residue_x = F.dropout(residue_x, block.dropout, training=self.training)
            res_feas.append(segment_csr(residue_x, node_ptr, reduce="max"))

            s_node = torch.softmax(_cluster(self.cluster[idx], residue_x, g), dim=-1)
            s = _dense(s_node, batch_size, max_res, res_idx)
            residue_hx = _dense(residue_x, batch_size, max_res, res_idx)
            cluster_x = self.norm(torch.bmm(s.transpose(1, 2), residue_hx))
            # MinCut loss: trace(S^T A S) over the edges and trace(S^T D S) over the residues.
            cut = segment_csr((s_node[g.src] * s_node[g.dst]).sum(-1), g.graph_edge_ptr, reduce="sum")
            volume = segment_csr(g.deg_out * s_node.square().sum(-1), node_ptr, reduce="sum")
            cluster_loss += (-(cut / volume)).mean()

            atom_x = F.relu(self.atom_update[idx](atom_x))
            mol_x, _ = self.mol_pools[idx](atom_x, mol_batch, batch_size)
            mol_x = self.norm(mol_x)
            mol_x = torch.cat([mol_total_fea, mol_x], dim=-1)
            mol_x = F.relu(self.atom_embed_total[idx](mol_x))
            mol_x = self.norm(self.atom_embed_total2[idx](mol_x))

            cluster_x = cluster_x.reshape(batch_size * num_cluster, -1)
            cluster_residue_batch = torch.arange(batch_size, device=device).repeat_interleave(
                num_cluster, output_size=batch_size * num_cluster
            )
            p2m_edge_index = torch.stack([torch.arange(batch_size * num_cluster, device=device), cluster_residue_batch])

            mol_x, cluster_x, inter_attn = self.inter_convs[idx](mol_x, cluster_x, p2m_edge_index)
            mol_feas.append(mol_x)
            inter_attn = inter_attn[1]

            atom_x = atom_x + F.relu(self.mol_update[idx](mol_x)[mol_batch])
            cluster_score = softmax(self.inter_attn_lin(inter_attn), cluster_residue_batch, num_nodes=batch_size)
            cluster_ptr = torch.arange(batch_size + 1, device=device) * num_cluster
            cluster_feas.append(self.norm(segment_csr(cluster_x * cluster_score, cluster_ptr, reduce="max")))

            cluster_hx = cluster_x.view(batch_size, num_cluster, -1)
            inter_attn = inter_attn.view(batch_size, num_cluster, -1)

            residue_x = residue_x + F.relu(self.res_update[idx]((s @ cluster_hx).flatten(0, 1)[res_idx]))
            residue_score = self.inter_attn_lin2((s @ inter_attn).flatten(0, 1)[res_idx])
            residue_score = softmax(residue_score, None, ptr=node_ptr)
            res_feas2.append(self.norm(segment_csr(residue_x * residue_score, node_ptr, reduce="max")))

        mol_x = F.relu(self.mol_fea_update(torch.cat(mol_feas, dim=-1)))
        res_feas = F.relu(self.res_fea_update(torch.cat(res_feas, dim=-1)))
        res_feas2 = F.relu(self.res_fea_update2(torch.cat(res_feas2, dim=-1)))
        clu_fea = F.relu(self.cluster_fea_update2(torch.cat(cluster_feas, dim=-1)))
        mol_prot_feat = torch.cat([res_feas, res_feas2, clu_fea, mol_x], dim=-1)

        reg_pred = F.relu(self.classifier(mol_prot_feat))
        reg_pred = F.relu(self.classifier1(reg_pred))
        return self.classifier2(reg_pred), cluster_loss
