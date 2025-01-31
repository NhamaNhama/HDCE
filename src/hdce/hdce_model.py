import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_context_encoder import LocalContextEncoder
from .global_semantic_module import GlobalSemanticModule
from .knowledge_interface import KnowledgeInterface
from .time_series_memory import TimeSeriesMemory

class HDCEModel(nn.Module):
    def __init__(self,
                 transformer_model_name,
                 embed_dim,
                 num_relations,
                 gnn_num_layers,
                 knowledge_graph_file,
                 decay_lambda=0.5,
                 kl_threshold=0.1):
        super().__init__()
        # テキストの局所文脈をエンコード
        self.local_encoder = LocalContextEncoder(transformer_model_name)
        # GNNを用いたグローバル文脈モジュール
        self.global_semantic = GlobalSemanticModule(embed_dim, num_relations, gnn_num_layers)
        # 知識グラフとのインターフェース
        self.knowledge_interface = KnowledgeInterface(knowledge_graph_file)
        # 時間的な分布のメモリ保持用モジュール
        self.memory = TimeSeriesMemory(embed_dim, decay_lambda, kl_threshold)
        self.embed_dim = embed_dim
    
    def forward(self, input_texts, graph_data, external_concept=None):
        # ローカルエンコーダから出力を得る
        local_out, attn_weights, attention_mask = self.local_encoder(input_texts)
        # BERTの[CLS]にあたるトークン表現を取り出す (B, D)
        cls_repr = local_out[:, 0, :]

        # グラフデータの特徴を取得
        node_x = graph_data["node_feature"]
        edge_index = graph_data["edge_index"]
        edge_type = graph_data["edge_type"]
        path_length = graph_data["path_length"]
        num_nodes = graph_data["num_nodes"]

        # グローバルなGNN処理を実行
        updated_nodes = self.global_semantic(
            node_x, edge_index, edge_type, path_length, num_nodes
        )
        # 何らかの形でCLS表現をノード表現にフィードバック
        updated_nodes[0] += cls_repr.mean(dim=0)

        knowledge_info = None
        if external_concept is not None:
            # 外部コンセプトを用いて確率的に知識グラフを探索
            knowledge_info = self.knowledge_interface.probabilistic_graph_walk(
                external_concept, steps=2, branching=2
            )
        
        # cls_reprをsoftmax分布にする
        distribution = F.softmax(cls_repr, dim=-1)
        # メモリモジュールを更新
        updated_dist, changed = self.memory(distribution)

        return {
            "local_out": local_out,
            "attn_weights": attn_weights,
            "updated_nodes": updated_nodes,
            "knowledge_info": knowledge_info,
            "updated_distribution": updated_dist,
            "significant_change": changed
        } 