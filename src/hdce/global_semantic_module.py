import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class RelationAwareGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations):
        # 親クラスMessagePassingの初期化（集約手法 'add' を指定）
        super().__init__(aggr='add')
        # 関係の種類数
        self.num_relations = num_relations
        # 関係ごとの重みパラメータ (num_relations, in_channels, out_channels)
        self.rel_weight = nn.Parameter(
            torch.randn(num_relations, in_channels, out_channels)
        )
        # 重みをXavierで初期化
        nn.init.xavier_uniform_(self.rel_weight)
        # バイアス
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_type, path_length):
        # メッセージ伝播→集約を行い、最終的な出力をバイアスと足して返す
        out = self.propagate(edge_index, x=x, edge_type=edge_type, path_length=path_length)
        out = out + self.bias
        return out

    def message(self, x_j, edge_type, path_length):
        # edge_typeに対応する関係の重みを取り出す
        w_rel = self.rel_weight[edge_type]  # (E, in_channels, out_channels)
        # x_j: 送信元ノードの特徴。einsumで行列積を計算
        out_msg = torch.einsum('eci,eio->eo', x_j, w_rel)
        # 経路長(path_length)でスケーリング（減衰）させる
        decayed = out_msg / (path_length.unsqueeze(-1).clamp_min(1e-9))
        return decayed

    def update(self, aggr_out):
        # 集約後の出力にReLUを適用
        return F.relu(aggr_out)

class GlobalSemanticModule(nn.Module):
    def __init__(self, embed_dim, num_relations, num_layers=2):
        super().__init__()
        # RelationAwareGNNConvを複数層積む
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelationAwareGNNConv(embed_dim, embed_dim, num_relations))

    def forward(self, x, edge_index, edge_type, path_length, num_nodes):
        # ノードへの自己ループを追加 (必要なら)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # GNNを多層適用
        for conv in self.convs:
            x = conv(x, edge_index, edge_type, path_length)
        return x 