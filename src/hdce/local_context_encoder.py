import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class ContextGateAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # 衝突スコア(conflict_score)を出すための小さなMLP
        self.conflict_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, Q, K, V, attention_mask=None):
        # スケーリングド・ドット積
        d_k = Q.size(-1)
        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / (d_k ** 0.5)
        B, L, _ = Q.shape
        
        # Q, Kをそれぞれ拡張して結合し、衝突スコアを計算
        Q_expand = Q.unsqueeze(2).expand(B, L, L, d_k)
        K_expand = K.unsqueeze(1).expand(B, L, L, d_k)
        combined = torch.cat([Q_expand, K_expand], dim=-1)  # (B, L, L, 2*d_k)
        
        conflict_score = self.conflict_proj(combined).squeeze(-1)  # (B, L, L)
        # ゲートとしてシグモイドを適用
        M_gate = torch.sigmoid(conflict_score)
        
        # 元のattentionスコアにゲートを掛ける
        gated_score = attn_score * M_gate
        
        # マスクがあれば適用
        if attention_mask is not None:
            gated_score = gated_score.masked_fill(attention_mask == 0, float('-inf'))
        
        # ソフトマックスで重みを正規化
        attn_weights = F.softmax(gated_score, dim=-1)
        # Vと掛け合わせる
        out = torch.matmul(attn_weights, V)
        return out, attn_weights

class LocalContextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        # Transformersライブラリを用いてトークナイザとモデルを読み込む
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        embed_dim = self.transformer.config.hidden_size
        # カスタムアテンション層
        self.context_gate_attn = ContextGateAttention(embed_dim)

    def forward(self, input_texts):
        # トークナイズ
        encoding = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        
        # Transformer本体の出力を取得
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        
        B, L, D = hidden_states.shape
        # attention_mask: (B, L)
        attn_mask_2d = attention_mask.unsqueeze(1).expand(-1, L, -1)
        
        # カスタムアテンションを適用
        local_out, attn_weights = self.context_gate_attn(hidden_states, hidden_states, hidden_states, attn_mask_2d)
        return local_out, attn_weights, attention_mask 