from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor
from typing import Literal
from .loss import LossCfg, Loss


# -------------------------------------
@dataclass
class LossGeomSemContrastCfg(LossCfg):
    name: Literal["geom_sem_contrast"] = "geom_sem_contrast"
    temperature: float = 0.1
    sample_k: int = 4096  # 每张图采样的像素点数量
    chunk_size: int | None = 2048 # InfoNCE 分块计算大小，用于节省显存


# 通用 InfoNCE
# -------------------------------------
class InfoNCECoreLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, chunk_size: int | None = None):
        super().__init__()
        # 对比学习温度参数
        self.temp = temperature
        # 分块计算大小时，用于节省显存
        self.chunk_size = chunk_size

    def forward(self, z_anchor: Tensor, z_positive: Tensor) -> Tensor:
        """
        计算 InfoNCE 损失。
        z_anchor   : [N, D] (N 对正样本，N 锚点)
        z_positive : [N, D] (N 对正样本，N 正例)
        InfoNCE 损失计算 N 个正样本对 (anchor_i, positive_i)
        并使用所有 N 个 positive 作为负样本。
        """
        # L2 归一化是对比学习的通用要求
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)

        N = z_anchor.shape[0]
        device = z_anchor.device
        
        # 标签：对角线是正样本 (0 vs 0, 1 vs 1, ...)
        labels = torch.arange(N, device=device)

        # 全量计算 (N <= chunk_size)
        if self.chunk_size is None or self.chunk_size >= N:
            # logits: [N_anchor, N_positive]
            logits = torch.mm(z_anchor, z_positive.t()) / self.temp
            return F.cross_entropy(logits, labels)

        # 分块计算 (当 N 很大时，避免 O(N^2) 矩阵乘法导致 OOM)
        losses = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            
            # 只计算 Anchor chunk 对 所有 Positive 的相似度
            logits_chunk = torch.mm(z_anchor[start:end], z_positive.t()) / self.temp
            
            # 正样本标签依然是 [0, 1, 2, ...] 上的子集
            losses.append(F.cross_entropy(logits_chunk, labels[start:end]))

        return torch.stack(losses).mean()


# -------------------------------------
# 主损失：几何 + 语义跨视角 InfoNCE (全视角对比实现)
# -------------------------------------
class GeomSemContrastLoss(Loss):
    def __init__(self, cfg: LossGeomSemContrastCfg) -> None:
        super().__init__(cfg)
        self.sample_k = cfg.sample_k
        self.core_loss = InfoNCECoreLoss(cfg.temperature, cfg.chunk_size)

    def _sample_vectors(self, feat: Tensor, idx: Tensor) -> Tensor:
        """
        辅助函数：根据固定的 idx 从特征图中提取向量
        feat: [B', C, H, W], 其中 B' = B * V
        idx:  [K] (在 H*W 范围内的索引)
        output: [B'*K, C]
        """
        # B', C, H, W = feat.shape
        # flat: [B', C, H*W]
        z = feat.flatten(2)
        # select: [B', C, K] -> permute -> [B', K, C]
        z = torch.index_select(z, 2, idx).permute(0, 2, 1)
        # flatten batch: [B'*K, C]
        return z.flatten(0, 1)

    def unweighted_loss(self, pred, gt) -> Float[Tensor, ""]:
        # 必要属性检查（先确保 pred 有 geo/sem 并且非 None）
        if not hasattr(pred, "geo") or not hasattr(pred, "sem") or pred.geo is None or pred.sem is None:
            # safe fallback: 返回标量 0 在 CPU（不会崩溃）
            return torch.tensor(0.0, device="cpu")

        geo = pred.geo
        sem = pred.sem

        # geo: [B, V, Cg, H, W]
        # sem: [B, V, Cs, H, W]
        B, V, Cg, H, W = geo.shape
        _, _, Cs, _, _ = sem.shape

        if V < 2:
            return torch.tensor(0.0, device=geo.device)

        # 随机像素采样（共享索引）
        K = min(self.sample_k, H * W)
        idx = torch.randint(0, H * W, (K,), device=geo.device)

        # 合并 B 和 V 方便采样：[B*V, C, H, W]
        geo_flat = geo.flatten(0, 1)
        sem_flat = sem.flatten(0, 1)

        # 采样 K 个点 -> [B*V*K, C]
        z_geo_flat = self._sample_vectors(geo_flat, idx)
        z_sem_flat = self._sample_vectors(sem_flat, idx)

        # reshape 回 [B, V, K, Cg] / [B, V, K, Cs]
        z_geo = z_geo_flat.reshape(B, V, K, Cg)
        z_sem = z_sem_flat.reshape(B, V, K, Cs)

        # 在 device 上初始化累加器，避免 device mismatch
        device = geo.device
        total_loss_geo = torch.tensor(0.0, device=device)
        total_loss_sem = torch.tensor(0.0, device=device)

        num_pairs = V * (V - 1)
        for i in range(V):
            for j in range(V):
                if i == j:
                    continue

                # 每对的 Anchor / Positive -> [B*K, C]
                z_geo_anchor = z_geo[:, i].flatten(0, 1)
                z_geo_positive = z_geo[:, j].flatten(0, 1)

                z_sem_anchor = z_sem[:, i].flatten(0, 1)
                z_sem_positive = z_sem[:, j].flatten(0, 1)

                total_loss_geo += self.core_loss(z_geo_anchor, z_geo_positive)
                total_loss_sem += self.core_loss(z_sem_anchor, z_sem_positive)

        loss_geo = total_loss_geo / num_pairs
        loss_sem = total_loss_sem / num_pairs
        return (loss_geo + loss_sem) * 0.5