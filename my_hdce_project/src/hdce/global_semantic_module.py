import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class RelationAwareGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__(aggr='add')
        self.num_relations = num_relations
        self.rel_weight = nn.Parameter(
            torch.randn(num_relations, in_channels, out_channels)
        )
        nn.init.xavier_uniform_(self.rel_weight)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_type, path_length):
        out = self.propagate(edge_index, x=x, edge_type=edge_type, path_length=path_length)
        out = out + self.bias
        return out

    def message(self, x_j, edge_type, path_length):
        w_rel = self.rel_weight[edge_type]  # (E, in_channels, out_channels)
        out_msg = torch.einsum('eci,eio->eo', x_j, w_rel)
        decayed = out_msg / (path_length.unsqueeze(-1).clamp_min(1e-9))
        return decayed

    def update(self, aggr_out):
        return F.relu(aggr_out)

class GlobalSemanticModule(nn.Module):
    def __init__(self, embed_dim, num_relations, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelationAwareGNNConv(embed_dim, embed_dim, num_relations))

    def forward(self, x, edge_index, edge_type, path_length, num_nodes):
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        for conv in self.convs:
            x = conv(x, edge_index, edge_type, path_length)
        return x 