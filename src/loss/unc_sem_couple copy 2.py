import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from typing import Literal

from .loss import Loss, LossCfg   # ⬅ 必须继承 Loss，而不是 nn.Module

# -------------------------------
@dataclass
class LossUncSemCoupleCfg(LossCfg):
    name: Literal["unc_sem_couple"] = "unc_sem_couple"
    reduction: str = "mean"

# -------------------------------
class UncSemCoupleLoss(Loss):   # ⬅ 继承 Loss
    """
    Uncertainty–Semantics Coupling Loss:
    L = sigma^2 * (1 - normalized(|z_sem|))
    """

    def __init__(self, cfg: LossUncSemCoupleCfg):
        super().__init__(cfg)   # ⬅ 必须调用 Loss.__init__
        self.reduction = getattr(cfg, "reduction", "mean")

    # ⚠⚠⚠ 框架要求实现这个函数，而不是 forward
    def unweighted_loss(self, pred, gt) -> Float[Tensor, ""]:
        """
        pred 里应该包含：
            pred.z_sem  : [B, C, H, W]
            pred.sigma  : [B, 1, H, W]
        gt 不使用（和 MSE / LPIPS 一样）
        """
        z_sem = pred.z_sem
        sigma = pred.sigma

        # 1. L2 magnitude
        magnitude = torch.linalg.norm(z_sem, dim=1)  # [B,H,W]

        # 2. normalize
        max_val = magnitude.amax(dim=[1,2], keepdim=True)
        max_val[max_val == 0] = 1e-6
        normalized = magnitude / max_val

        # 3. (1 - norm)
        coupling = 1.0 - normalized

        # 4. sigma^2
        sigma_sq = sigma.squeeze(1).pow(2)

        # 5. pixel loss map
        loss_map = sigma_sq * coupling

        # 6. reduction
        if self.reduction == "mean":
            return loss_map.mean()
        elif self.reduction == "sum":
            return loss_map.sum()
        else:
            return loss_map
