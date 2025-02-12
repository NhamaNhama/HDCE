import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesMemory(nn.Module):
    def __init__(self, embed_dim=768, decay_lambda=0.5, kl_threshold=0.1):
        super().__init__()
        self.prev_distribution = None
        self.embed_dim = embed_dim
        self.decay_lambda = decay_lambda
        self.kl_threshold = kl_threshold
    
    def kl_divergence(self, p, q):
        # p, q のKLダイバージェンスを計算
        p = p.clamp_min(1e-9)
        q = q.clamp_min(1e-9)
        return (p * (p.log() - q.log())).sum(dim=-1)
    
    def time_decay_integration(self, old_repr, new_repr):
        # 指数的減衰を用いて古い表現と新しい表現を統合
        w = torch.exp(-self.decay_lambda)
        return w * old_repr + (1 - w) * new_repr
    
    def forward(self, distribution):
        # 前回の分布がなければ初回なのでそのまま保持
        if self.prev_distribution is None:
            self.prev_distribution = distribution
            return distribution, False
        
        # 今回の分布と前回の分布のKLダイバージェンスを算出
        kl_vals = self.kl_divergence(distribution, self.prev_distribution)
        avg_kl = kl_vals.mean()
        changed = bool(avg_kl.item() > self.kl_threshold)
        
        # 減衰統合
        updated_repr = self.time_decay_integration(self.prev_distribution, distribution)
        
        # KL差が大きければprev_distributionを更新、そうでなければ減衰後の表現を使う
        if changed:
            self.prev_distribution = distribution
        else:
            self.prev_distribution = updated_repr
        
        return self.prev_distribution, changed 