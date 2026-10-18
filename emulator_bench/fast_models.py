"""Faster KcatNet with the same parameters and outputs as models.model_kcat.KcatNet.

The subclasses avoid per-tower indexing, host-to-device copies and host synchronisations.
Batch sizes and residue indices come from the CPU batch, so the forward pass never reads GPU tensors.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.utils import softmax, to_dense_adj

from models.model_kcat import KcatNet
from models.Mol_pool import MotifPool
from models.pna import PNAConv
from models.protein_pool import dense_mincut_pool


class FastPNAConv(PNAConv):
    def forward(self, x, edge_index, edge_attr=None):
        x = x.view(-1, self.towers, self.F_in)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = torch.cat([x, out], dim=-1)
        out = torch.cat([nn(o) for o, nn in zip(out.unbind(1), self.post_nns)], dim=1)
        return self.lin(out)

    def message(self, x_i, x_j, edge_attr):
        edge_attr = self.edge_encoder(edge_attr).view(-1, 1, self.F_in).repeat(1, self.towers, 1)
        h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return torch.stack([nn(o) for o, nn in zip(h.unbind(1), self.pre_nns)], dim=1)


class FastMotifPool(MotifPool):
    def forward(self, x, mol_batch, batch_size=None):
        H, C = self.heads, self.hidden_dim
        atom_feature = x.view(-1, H, C)
        score = torch.cat(
            [mlp(a) for a, mlp in zip(atom_feature.unbind(1), self.score_proj)], dim=-1
        )
        alpha = softmax(score, mol_batch, num_nodes=batch_size)
        drug_feat = global_add_pool(
            (atom_feature * alpha.view(-1, H, 1)).view(-1, H * C), mol_batch, batch_size
        )
        return drug_feat, alpha


def _rbf(D, D_max, D_count):
    D_mu = torch.linspace(0.0, D_max, D_count, device=D.device).view(1, -1)
    return torch.exp(-(((D.clamp(max=D_max).unsqueeze(-1) - D_mu) / (D_max / D_count)) ** 2))


def _dense(x, batch_size, max_res, res_idx):
    out = x.new_zeros(batch_size * max_res, x.size(-1))
    out[res_idx] = x
    return out.view(batch_size, max_res, -1)


class FastKcatNet(KcatNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Switching the class keeps the parameters and their initialisation unchanged.
        for block in self.prot_convs:
            block.conv.__class__ = FastPNAConv
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
        batch_size,
        max_res,
        res_mask,
        res_idx,
    ):
        """`res_mask` is the (B, max_res) residue mask and `res_idx` its flat nonzero indices."""
        device = residue_esm.device
        cluster_loss = torch.zeros((), device=device)
        residue_scores = []

        residue_ini = torch.cat(
            [F.relu(self.seq_embed_prot5(residue_prot5)), F.relu(self.seq_embed_esm(residue_esm))],
            dim=-1,
        )
        residue_x = F.relu(self.seq_embed(residue_ini))
        residue_edge_attr = _rbf(residue_edge_weight, 1.0, self.prot_edge_dim)

        atom_x = self.atom_type_embed(mol_x.squeeze()) + F.relu(self.atom_feat_embed(mol_x_feat))
        mol_total_fea = F.relu(self.mol_embed(mol_total_fea))
        mol_total_fea = self.norm(self.mol_embed2(mol_total_fea))

        residue_adj0 = to_dense_adj(
            residue_edge_index, prot_batch, max_num_nodes=max_res, batch_size=batch_size
        )
        res_feas, res_feas2, mol_feas, cluster_feas = [], [], [], []

        for idx in range(self.total_layer):
            num_cluster = self.num_cluster[idx]
            residue_x = self.GN(residue_x, prot_batch, batch_size)
            atom_x = self.GN(atom_x, mol_batch, batch_size)

            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)
            res_feas.append(global_max_pool(residue_x, prot_batch, batch_size))

            s = _dense(
                self.cluster[idx](residue_x, residue_edge_index), batch_size, max_res, res_idx
            )
            residue_hx = _dense(residue_x, batch_size, max_res, res_idx)
            s, cluster_x, _, cl_loss, _ = dense_mincut_pool(
                residue_hx, residue_adj0, s, res_mask, None
            )
            cluster_x = self.norm(cluster_x)
            cluster_loss += cl_loss

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
            p2m_edge_index = torch.stack(
                [torch.arange(batch_size * num_cluster, device=device), cluster_residue_batch]
            )

            mol_x, cluster_x, inter_attn = self.inter_convs[idx](mol_x, cluster_x, p2m_edge_index)
            mol_feas.append(mol_x)
            inter_attn = inter_attn[1]

            atom_x = atom_x + F.relu(self.mol_update[idx](mol_x)[mol_batch])
            cluster_score = softmax(
                self.inter_attn_lin(inter_attn), cluster_residue_batch, num_nodes=batch_size
            )
            cluster_feas.append(
                self.norm(
                    global_max_pool(cluster_x * cluster_score, cluster_residue_batch, batch_size)
                )
            )

            cluster_hx = cluster_x.view(batch_size, num_cluster, -1)
            inter_attn = inter_attn.view(batch_size, num_cluster, -1)

            residue_x = residue_x + F.relu(
                self.res_update[idx]((s @ cluster_hx).flatten(0, 1)[res_idx])
            )
            residue_score = self.inter_attn_lin2((s @ inter_attn).flatten(0, 1)[res_idx])
            residue_score = softmax(residue_score, prot_batch, num_nodes=batch_size)
            residue_scores.append(residue_score)
            res_feas2.append(
                self.norm(global_max_pool(residue_x * residue_score, prot_batch, batch_size))
            )

        mol_x = F.relu(self.mol_fea_update(torch.cat(mol_feas, dim=-1)))
        res_feas = F.relu(self.res_fea_update(torch.cat(res_feas, dim=-1)))
        res_feas2 = F.relu(self.res_fea_update2(torch.cat(res_feas2, dim=-1)))
        clu_fea = F.relu(self.cluster_fea_update2(torch.cat(cluster_feas, dim=-1)))
        mol_prot_feat = torch.cat([res_feas, res_feas2, clu_fea, mol_x], dim=-1)

        reg_pred = F.relu(self.classifier(mol_prot_feat))
        reg_pred = F.relu(self.classifier1(reg_pred))
        return self.classifier2(reg_pred), cluster_loss
