# src/loss/geom_sem_contrast.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor
from .loss import LossCfg, Loss

# ----------- 配置 -----------
@dataclass
class LossGeomSemContrastCfg(LossCfg):
    temperature: float = 0.1
    sample_k: int = 2048
    chunk_size: int = 1024

# ----------- 核心实现 -----------
class InfoNCECoreLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, chunk_size: int | None = None):
        super().__init__()
        self.temp = temperature
        self.chunk_size = chunk_size

    def forward(self, z: Tensor, z_pos: Tensor) -> Tensor:
        """
        z     : [N, D] 查询
        z_pos : [N, D] 正样本
        做 self+ 正样本对比，标签 arange(N)
        """
        z = F.normalize(z, dim=-1)
        z_pos = F.normalize(z_pos, dim=-1)
        N = z.shape[0]
        labels = torch.arange(N, device=z.device)

        if self.chunk_size is None or self.chunk_size >= N:
            logits = torch.mm(z, z_pos.t()) / self.temp
            return F.cross_entropy(logits, labels)

        losses = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            logits_chunk = torch.mm(z[start:end], z_pos.t()) / self.temp
            losses.append(F.cross_entropy(logits_chunk, labels[start:end]))
        return torch.stack(losses).mean()

# ----------- 正式损失 -----------
class GeomSemContrastLoss(Loss):
    def __init__(self, cfg: LossGeomSemContrastCfg) -> None:
        super().__init__(cfg)
        self.sample_k = cfg.sample_k
        self.core_loss = InfoNCECoreLoss(cfg.temperature, cfg.chunk_size)

    def unweighted_loss(self, pred, gt) -> Float[Tensor, ""]:
        feat = pred.image  # 可能是 [B, V, C, H, W]
        if feat is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # 统一降到 4 维：取 view-0
        if feat.dim() == 5:
            feat = feat[:, 0]  # [B, C, H, W]

        B, C, H, W = feat.shape
        K = min(self.sample_k, H * W)
        idx = torch.randint(0, H * W, (K,), device=feat.device)

        dim = min(32, C)  # 防越界
        z = feat[:, :dim].flatten(2)  # [B, dim, H*W]
        z = torch.index_select(z, 2, idx).permute(0, 2, 1)  # [B, K, dim]
        z = z.flatten(0, 1)  # [B*K, dim]

        # 自对比
        loss = self.core_loss(z, z)
        return loss