"""
GNN for graph classification. Returns logits and (optionally) node features + graph embedding for PMH.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


def get_model(num_node_features, num_classes, hidden=64, num_layers=4):
    return GraphGCN(
        num_node_features=num_node_features,
        num_classes=num_classes,
        hidden=hidden,
        num_layers=num_layers,
    )


class GraphGCN(nn.Module):
    """
    GCN stack + global mean pool + classifier.
    forward(x, edge_index, batch, return_embeddings=True) returns (logits, node_feats, graph_emb).
    """
    def __init__(self, num_node_features, num_classes, hidden=64, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch, return_embeddings=False, return_all_layers=False):
        node_feats = x
        all_graph_embs = []
        for i in range(self.num_layers):
            node_feats = self.convs[i](node_feats, edge_index)
            node_feats = self.bns[i](node_feats)
            node_feats = node_feats.relu()
            if return_all_layers:
                all_graph_embs.append(global_mean_pool(node_feats, batch))
        graph_emb = all_graph_embs[-1] if return_all_layers else global_mean_pool(node_feats, batch)
        logits = self.classifier(graph_emb)
        if return_all_layers:
            # all_graph_embs: list of [B, hidden] at each GNN depth level
            return logits, node_feats, graph_emb, all_graph_embs
        if return_embeddings:
            return logits, node_feats, graph_emb
        return logits
