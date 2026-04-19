"""
GNN for multi-task molecular regression (QM9). One shared encoder, 19 outputs.
Uses both node features (x) and 3D positions (pos). Returns (pred [B,19], node_emb, graph_emb) for PMH.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

NUM_TARGETS = 19


def get_model(num_node_features, num_targets=NUM_TARGETS, hidden=128, num_layers=4):
    return MolGCN(
        num_node_features=num_node_features,
        num_targets=num_targets,
        hidden=hidden,
        num_layers=num_layers,
    )


class MolGCN(nn.Module):
    """
    Geometry-aware GNN: embed x and pos, then GCN + global mean pool -> 19 targets.
    forward(..., return_embeddings=True) returns (pred [B,19], node_emb, graph_emb).
    """

    def __init__(self, num_node_features, num_targets=NUM_TARGETS, hidden=128, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.input_lin = nn.Linear(num_node_features + 3, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.predictor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_targets),
        )

    def forward(self, x, pos, edge_index, batch, return_embeddings=False):
        if x.dtype != torch.float32:
            x = x.float()
        if pos is not None and pos.dtype != torch.float32:
            pos = pos.float()
        node_feats = torch.cat([x, pos], dim=-1)
        node_feats = F.relu(self.input_lin(node_feats))
        for i in range(self.num_layers):
            node_feats = self.convs[i](node_feats, edge_index)
            node_feats = self.bns[i](node_feats)
            node_feats = node_feats.relu()
        graph_emb = global_mean_pool(node_feats, batch)
        pred = self.predictor(graph_emb)  # [B, num_targets]
        if return_embeddings:
            return pred, node_feats, graph_emb
        return pred
