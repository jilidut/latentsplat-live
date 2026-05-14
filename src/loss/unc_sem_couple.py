# src/loss/unc_sem_couple.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor
from typing import Literal

from .loss import Loss, LossCfg


# -------------------------- 配置 --------------------------
@dataclass
class LossUncSemCoupleCfg(LossCfg):
    name: Literal["unc_sem_couple"] = "unc_sem_couple"
    enabled: bool = True 


# -------------------------- 主损失 --------------------------
class UncSemCoupleLoss(Loss):
    def __init__(self, cfg: LossUncSemCoupleCfg):
        super().__init__(cfg)
        self._cache: tuple[Tensor, Tensor] | None = None

    # ---- 外部注入 token ----
    def cache(self, sem: Tensor, unc: Tensor) -> None:
        self._cache = (sem.detach(), unc.detach())

    # ---- 必须实现的抽象方法 ----
    def unweighted_loss(self, pred, gt) -> Float[Tensor, ""]:
        if self._cache is not None:
            sem, unc = self._cache
            self._cache = None
            return self._token_loss(sem, unc)

        if hasattr(pred, "sem") and hasattr(pred, "unc") and pred.sem is not None and pred.unc is not None:
            return self._token_loss(pred.sem, pred.unc)

        device = pred.image.device if hasattr(pred, 'image') else 'cpu'
        return torch.tensor(0.0, device=device)

    def _token_loss(self, sem: Tensor, unc: Tensor) -> Tensor:
        """
        sem: [B, V, N, 32]
        unc: [B, V, N, 1]   —— log σ 形式，值通常在 [-3, 3]
        """

        B, V, N, _ = sem.shape
        device = sem.device

        # ---------- 1️⃣ 限制最大 token 数（稳定 & 节约显存） ----------
        max_tokens = 4096
        if N > max_tokens:
            idx = torch.randperm(N, device=device)[:max_tokens]
            sem = sem[..., idx, :]
            unc = unc[..., idx, :]

        # ---------- 2️⃣ merge 维度 ----------
        sem = sem.flatten(1, 2)      # [B, V*N, 32]
        unc = unc.flatten(1, 2)      # [B, V*N, 1]

        #临时打印
        if self.training and B == 2:
            print(f"[UNC-DEBUG] unc min={unc.min():.3f}, max={unc.max():.3f}, mean={unc.mean():.3f}")
        #临时打印

        # ---------- 3️⃣ 归一化，确保 sim ∈ [-1,1] ----------
        sem = F.normalize(sem, dim=-1)

        # ---------- 4️⃣ cos 相似度 ----------
        sim = torch.bmm(sem, sem.transpose(1, 2))   # [B, M, M]

        # ---------- 5️⃣ 不确定性权重 ----------
        # unc 是 log σ
        # σ = exp(unc)，不确定性越大 → 惩罚越弱
        sigma = torch.exp(unc).clamp(0.1, 10.0)      # 数值安全
        weight = (1.0 / sigma).squeeze(-1)           # [B, M]

        # ---------- 6️⃣ loss ----------
        # InfoNCE-like：鼓励 sim→1，但不是暴力
        loss_map = weight[:, :, None] * (1 - sim)

        # ---------- 7️⃣ 稳定保险 ----------
        loss = loss_map.mean()
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

        return loss

    # ---- 单元测试入口 ----
    def test_forward(self, sem: Tensor, unc: Tensor) -> Tensor:
        return self._token_loss(sem, unc)

# -------------------------- 单元测试 --------------------------
if __name__ == "__main__":
    B, V, N = 2, 3, 256
    sem = torch.randn(B, V, N, 32)
    unc = torch.randn(B, V, N, 1)
    cfg = LossUncSemCoupleCfg()
    loss_fn = UncSemCoupleLoss(cfg)
    loss = loss_fn.test_forward(sem, unc)
    print("✅ UncSemCoupleLoss 入口正常，loss =", loss.item())
    #python -m src.loss.unc_sem_couple