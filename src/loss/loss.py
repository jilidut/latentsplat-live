from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from jaxtyping import Float
import torch
from torch import Tensor, nn

from ..model.types import Prediction, GroundTruth
#新增
from dataclasses import dataclass, field
from typing import Any
#新增
@dataclass
class LossCfg:
    name: str
    weight: float | int = 1
    apply_after_step: int = 0
    #新增
    temperature: float = 0.1
    sample_k: int = 2048
    chunk_size: int = 1024
    
    extra: dict[str, Any] = field(default_factory=dict)
    #新增


@dataclass
class LossValue:
    unweighted: Float[Tensor, ""]
    weighted: Float[Tensor, ""]


LossOutput = LossValue | Dict[str, 'LossOutput']    # will be flattened in LossGroup


class Loss(nn.Module, ABC):
    cfg: LossCfg
    name: str

    def __init__(self, cfg: LossCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = cfg.name

    @abstractmethod
    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth | None = None
    ) -> Float[Tensor, ""]:
        pass

    def forward(
        self,
        prediction: Prediction,
        gt: GroundTruth | None = None,
        global_step: int = 0
    ) -> LossOutput:
        
        print(f"[DEBUG] Loss '{self.name}' apply_after_step={self.cfg.apply_after_step}, global_step={global_step}")

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            unweighted = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        else:
            unweighted = self.unweighted_loss(prediction, gt)
        weighted = self.cfg.weight * unweighted
        return LossValue(unweighted, weighted)
    
    def is_active(self, global_step: int) -> bool:
        return self.cfg.apply_after_step <= global_step
