import torch
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from typing import Literal

from .loss import Loss, LossCfg

# ------------------------------- 
@dataclass
class LossUncSemCoupleCfg(LossCfg):
    name: Literal["unc_sem_couple"] = "unc_sem_couple"

# -------------------------------
class UncSemCoupleLoss(Loss):
    """
    不确定性-语义耦合损失:
    L = mean( |I_pred - I_gt| / sigma + log(sigma) )
    sigma: [B,1,H,W] 预测 logσ
    """

    def __init__(self, cfg: LossUncSemCoupleCfg):
        super().__init__(cfg)

    def unweighted_loss(self, pred, gt) -> Float[Tensor, ""]:
        rgb_pred = pred.rgb
        rgb_gt   = gt

        sigma = pred.sigma.exp().clamp(1e-3, 10.0)
        loss_map = (rgb_pred - rgb_gt).abs() / sigma + torch.log(sigma)
        return loss_map.mean()