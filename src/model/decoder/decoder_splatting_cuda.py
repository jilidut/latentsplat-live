# src/model/decoder/decoder_splatting_cuda.py
from dataclasses import dataclass
from typing import Literal
from dataclasses import replace

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, RenderOutput, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        background_color: list[float] = [0., 0., 0.],
        variational: bool = False
    ) -> None:
        super().__init__(cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.variational = variational

    def render_to_decoder_output(
        self,
        render_output: RenderOutput,
        b: int,
        v: int
    ) -> DecoderOutput:
        if render_output.feature is not None:
            features = rearrange(render_output.feature, "(b v) c h w -> b v c h w", b=b, v=v)
            mean, logvar = features.chunk(2, dim=2) if self.variational \
                else (features, (1-rearrange(render_output.mask.detach(), "(b v) h w -> b v () h w", b=b, v=v)).log().expand_as(features))
            feature_posterior = DiagonalGaussianDistribution(mean, logvar)
        else:
            feature_posterior = None
        return DecoderOutput(
            color=rearrange(render_output.color, "(b v) c h w -> b v c h w", b=b, v=v) if render_output.color is not None else None,
            feature_posterior=feature_posterior,
            mask=rearrange(render_output.mask, "(b v) h w -> b v h w", b=b, v=v),
            depth=rearrange(render_output.depth, "(b v) h w -> b v h w", b=b, v=v)
        )
    
   
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        return_colors: bool = True,
        return_features: bool = True
    ) -> DecoderOutput:
        

        b, v, _, _ = extrinsics.shape

        color_sh = repeat(gaussians.color_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) \
            if return_colors and gaussians.color_harmonics is not None else None
        
        feature_sh = repeat(gaussians.features, "b g c d_sh -> (b v) g c d_sh", v=v) \
            if return_features and hasattr(gaussians, 'features') and gaussians.features is not None else None
        

        # 确保所有张量在同一设备
        device = gaussians.means.device
        color_sh = color_sh.to(device)
        if feature_sh is not None:
            feature_sh = feature_sh.to(device)

        H, W = color_sh.shape[-2:]   # 安全写法

        print("Gaussians fields:", dir(gaussians))

        from dataclasses import replace

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 裁剪高斯
        max_gaussians = 1_000
        if gaussians.means.shape[0] > max_gaussians:
            idx = torch.randperm(gaussians.means.shape[0], device=device)[:max_gaussians]
            gaussians = replace(
                gaussians,
                means=gaussians.means[idx],
                covariances=gaussians.covariances[idx],
                opacities=gaussians.opacities[idx],
                color_harmonics=gaussians.color_harmonics[idx],
                feature_harmonics=gaussians.feature_harmonics[idx] if gaussians.feature_harmonics is not None else None,
            )

        # 2. 统一搬设备
        extrinsics = extrinsics.to(device)
        intrinsics = intrinsics.to(device)
        color_sh   = color_sh.to(device)
        if feature_sh is not None:
            feature_sh = feature_sh.to(device)

        gaussians = replace(
            gaussians,
            means=gaussians.means.to(device),
            covariances=gaussians.covariances.to(device),
            opacities=gaussians.opacities.to(device),
            color_harmonics=gaussians.color_harmonics.to(device),
            feature_harmonics=None,
        )

        # ---------- 安全获取 scale ----------
        if hasattr(gaussians, 'scales') and gaussians.scales is not None:
            scale = gaussians.scales
        elif hasattr(gaussians, 'scale') and gaussians.scale is not None:
            scale = gaussians.scale
        else:
            import logging
            logging.getLogger(__name__).warning(
                "Neither 'scales' nor 'scale' found in gaussians, falling back to torch.ones"
            )
            scale = torch.ones(gaussians.means.shape[0], device=gaussians.means.device)
        # ---------- scale 获取完毕 ----------

        # 5. 渲染
        rendered: RenderOutput = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            color_sh,
            feature_sh,
        )
        out = self.render_to_decoder_output(rendered, b, v)
        if depth_mode is not None and depth_mode != "depth":
            out.depth = self.render_depth(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode)
        return out

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
    
    from .cuda_splatting import render_cuda, RenderOutput

    def render_rgb(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "4 4"],  # 单个相机
        intrinsics: Float[Tensor, "3 3"],
        near: float,
        far: float,
        image_size: tuple[int, int],
    ) -> Float[Tensor, "height width 3"]:
        """单相机彩色渲染，供 GIF 回调使用"""
        # 升维到 (1,1,...) 再扔给已有 forward
        out: DecoderOutput = self(
            gaussians,
            extrinsics[None, None],
            intrinsics[None, None],
            torch.tensor([[near]], device=extrinsics.device),
            torch.tensor([[far]], device=extrinsics.device),
            image_size,
            return_colors=True,
            return_features=False,
        )
        # 取出颜色图并转到 HWC
        rgb = out.color[0, 0].permute(1, 2, 0).clamp(0, 1)  # (H,W,3)
        return rgb


    def last_layer_weights(self) -> None:
        return None
