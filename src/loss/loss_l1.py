from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ..model.types import Prediction, GroundTruth
from .loss import LossCfg, Loss


@dataclass
class LossL1Cfg(LossCfg):
    name: Literal["l1"] = "l1"
    weight: float = 1.0
    apply_after_step: int = 0

class LossL1(Loss):
    def __init__(self, cfg: LossL1Cfg):
        super().__init__(cfg)
        print(f"[INIT] LossL1 initialized with apply_after_step={self.cfg.apply_after_step}")

    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth
    ) -> Float[Tensor, ""]:
        # delta = prediction.image - gt.image

        if prediction is None or gt is None:
            return 0.0
        delta = prediction.image - gt.image
        return delta.abs().mean()
