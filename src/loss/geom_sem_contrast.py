# src/loss/geom_sem_contrast.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor
from typing import Literal

from .loss import Loss, LossCfg


# -------------------------- 配置 --------------------------
@dataclass
class LossGeomSemContrastCfg(LossCfg):
    name: Literal["geom_sem_contrast"] = "geom_sem_contrast"
    enabled: bool = True 
    temperature: float = 0.1
    sample_k: int = 4096          # 每张图采样像素数
    chunk_size: int | None = 2048  # InfoNCE 分块大小


# -------------------------- InfoNCE 核心 --------------------------
class InfoNCECoreLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, chunk_size: int | None = None):
        super().__init__()
        self.temp = temperature
        self.chunk_size = chunk_size

    def forward(self, z_anchor: Tensor, z_positive: Tensor) -> Tensor:
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)
        N = z_anchor.shape[0]
        device = z_anchor.device
        labels = torch.arange(N, device=device)

        if self.chunk_size is None or self.chunk_size >= N:
            logits = torch.mm(z_anchor, z_positive.t()) / self.temp
            return F.cross_entropy(logits, labels)

        losses = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            logits_chunk = torch.mm(z_anchor[start:end], z_positive.t()) / self.temp
            losses.append(F.cross_entropy(logits_chunk, labels[start:end]))
        return torch.stack(losses).mean()


# -------------------------- 主损失 --------------------------
class GeomSemContrastLoss(Loss):
    def __init__(self, cfg: LossGeomSemContrastCfg) -> None:
        super().__init__(cfg)
        self.sample_k = cfg.sample_k
        self.core_loss = InfoNCECoreLoss(cfg.temperature, cfg.chunk_size)
        self._cache: tuple[Tensor, Tensor] | None = None

    # ---- 外部注入 token ----
    def cache(self, geom: Tensor, sem: Tensor) -> None:
        self._cache = (geom.detach(), sem.detach())

    # ---- 唯一抽象方法 ----
    def unweighted_loss(self, pred, gt) -> Float[Tensor, ""]:
        # 1. 优先用缓存（training_step 已注入）
        if self._cache is not None:
            geom, sem = self._cache
            self._cache = None
            return self._token_loss(geom, sem)

        # 2. 无缓存，从 pred 取 token 级特征
        if hasattr(pred, "geo") and hasattr(pred, "sem") and pred.geo is not None and pred.sem is not None:
            return self._token_loss(pred.geo, pred.sem)   # 统一用 token 级对比

        # 3. 兜底零
        device = pred.image.device if hasattr(pred, 'image') else 'cpu'
        return torch.tensor(0.0, device=device)

    def _token_loss(self, geom: Tensor, sem: Tensor) -> Tensor:
        """
        geom: [B, V, N, C]
        sem : [B, V, N, C]
        """
        B, V, N, C = geom.shape
        device = geom.device

        # ---------- 1️⃣ 采样 ----------
        max_tokens = 4096
        if N > max_tokens:
            idx = torch.randperm(N, device=device)[:max_tokens]
            geom = geom[..., idx, :]
            sem  = sem[...,  idx, :]
            N = max_tokens

        # ---------- 2️⃣ flatten ----------
        geom = geom.flatten(0, 2)  # [B*V*N, C]
        sem  = sem.flatten(0, 2)   # [B*V*N, C]

        # ---------- 3️⃣ normalize ----------
        geom = F.normalize(geom, dim=-1)
        sem  = F.normalize(sem, dim=-1)

        # ---------- 4️⃣ logits ----------
        logits = torch.mm(geom, sem.t()) / self.cfg.temperature

        #临时打印
        if self.training and B == 2:
            print(f"[GEO-DEBUG] B={B}, V={V}, N={N}")
            print(f"[GEO-DEBUG] logits min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
        #临时打印

        # ---------- 5️⃣ target：对齐自身 ----------
        M = logits.shape[0]
        labels = torch.arange(M, device=device)
        loss = F.cross_entropy(logits, labels)
        return loss

    # ---- 像素采样逻辑（保留，但不再被调用）----
    def _sample_vectors(self, feat: Tensor, idx: Tensor) -> Tensor:
        z = feat.flatten(2)
        z = torch.index_select(z, 2, idx).permute(0, 2, 1)
        return z.flatten(0, 1)

    def _spatial_sample_loss(self, geo: Tensor, sem: Tensor) -> Tensor:
        B, V, Cg, H, W = geo.shape
        _, _, Cs, _, _ = sem.shape
        if V < 2:
            return torch.tensor(0.0, device=geo.device)

        K = min(self.sample_k, H * W)
        idx = torch.randint(0, H * W, (K,), device=geo.device)

        geo_flat = geo.flatten(0, 1)
        sem_flat = sem.flatten(0, 1)

        z_geo_flat = self._sample_vectors(geo_flat, idx)
        z_sem_flat = self._sample_vectors(sem_flat, idx)

        z_geo = z_geo_flat.reshape(B, V, K, Cg)
        z_sem = z_sem_flat.reshape(B, V, K, Cs)

        device = geo.device
        total_loss_geo = torch.tensor(0.0, device=device)
        total_loss_sem = torch.tensor(0.0, device=device)
        num_pairs = V * (V - 1)

        for i in range(V):
            for j in range(V):
                if i == j:
                    continue
                z_geo_anchor   = z_geo[:, i].flatten(0, 1)
                z_geo_positive = z_geo[:, j].flatten(0, 1)
                z_sem_anchor   = z_sem[:, i].flatten(0, 1)
                z_sem_positive = z_sem[:, j].flatten(0, 1)

                total_loss_geo += self.core_loss(z_geo_anchor, z_geo_positive)
                total_loss_sem += self.core_loss(z_sem_anchor, z_sem_positive)

        loss_geo = total_loss_geo / num_pairs
        loss_sem = total_loss_sem / num_pairs
        return (loss_geo + loss_sem) * 0.5

    # ---- 单元测试入口 ----
    def test_forward(self, geom: Tensor, sem: Tensor) -> Tensor:
        return self._token_loss(geom, sem)

# -------------------------- 单元测试 --------------------------
if __name__ == "__main__":
    B, V, N, C = 2, 3, 256, 32
    geom = torch.randn(B, V, N, C)
    sem  = torch.randn(B, V, N, C)
    cfg = LossGeomSemContrastCfg()
    loss_fn = GeomSemContrastLoss(cfg)
    loss = loss_fn.test_forward(geom, sem)
    print("✅ GeomSemContrastLoss 入口正常，loss =", loss.item())
# python -m src.loss.geom_sem_contrast