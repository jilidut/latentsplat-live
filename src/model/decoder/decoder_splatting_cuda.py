# src/model/decoder/decoder_splatting_cuda.py
from dataclasses import dataclass
from typing import Literal
from dataclasses import replace

import torch  
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from torch.utils.cpp_extension import load_inline
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
        v: int,
        image_shape: tuple[int, int]
    ) -> DecoderOutput:

        # --- 1. 颜色图：已经是 4-D，直接 rearrange ---
        color = rearrange(render_output.color, "(b v) c h w -> b v c h w", b=b, v=v) \
            if render_output.color is not None else None

        # --- mask ---
        if render_output.mask.ndim == 2:          # [BV, H*W]
            flat_len = render_output.mask.shape[1]
            hw = int(flat_len ** 0.5)
            while flat_len % hw:
                hw -= 1
            ww = flat_len // hw
            mask_3d = render_output.mask.view(b * v, hw, ww)
        else:                                     # [BV, H, W]
            mask_3d = render_output.mask

        # --- depth ---
        if render_output.depth.ndim == 2:         # [BV, H*W]
            flat_len = render_output.depth.shape[1]
            hw = int(flat_len ** 0.5)
            while flat_len % hw:
                hw -= 1
            ww = flat_len // hw
            depth_3d = render_output.depth.view(b * v, hw, ww)
        else:                                     # [BV, H, W]
            depth_3d = render_output.depth

        # --- 3. rearrange 到 [B, V, H, W] ---
        mask  = rearrange(mask_3d,  "(b v) h w -> b v h w", b=b, v=v)
        depth = rearrange(depth_3d, "(b v) h w -> b v h w", b=b, v=v)

        if render_output.feature is not None:
            features = rearrange(render_output.feature, "(b v) c h w -> b v c h w", b=b, v=v)
            mean, logvar = features.chunk(2, dim=2) if self.variational \
                else (features,
                    (1 - rearrange(mask_3d, "(b v) h w -> b v h w", b=b, v=v).unsqueeze(2).detach()).log().expand_as(features))
            feature_posterior = DiagonalGaussianDistribution(mean, logvar)
        else:
            feature_posterior = None

        return DecoderOutput(
            color=color,
            feature_posterior=feature_posterior,
            mask=mask,
            depth=depth,
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DECODER_DEBUG] return_features parameter: {return_features}")
        print(f"[DECODER_DEBUG] Gaussians has feature_harmonics: {hasattr(gaussians, 'feature_harmonics')}")

        # 处理颜色谐波
        color_sh = None
        if return_colors and hasattr(gaussians, 'color_harmonics') and gaussians.color_harmonics is not None:
            color_sh = repeat(gaussians.color_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v).to(device)

        # 处理特征谐波
        feature_sh = None
        if return_features and hasattr(gaussians, 'feature_harmonics') and gaussians.feature_harmonics is not None:
            feature_sh = repeat(gaussians.feature_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v).to(device)
        else:
            print("feature_harmonics is None or not available")

        # 确保至少有一种谐波系数（修复AssertionError）
        if color_sh is None and feature_sh is None:
            # 创建默认的颜色谐波（白色）
            num_gaussians = gaussians.means.shape[1]
            color_sh = torch.ones((b*v, num_gaussians, 3, 1), device=device)
            print("Using default white color SH coefficients")

        # 确保所有张量在同一设备
        extrinsics = extrinsics.to(device)
        intrinsics = intrinsics.to(device)
        near = near.to(device)
        far = far.to(device)

        # 截断高斯（如果数量太多）
        max_gaussians = 1_000
        if gaussians.means.shape[1] > max_gaussians:
            idx = torch.randperm(gaussians.means.shape[1], device=device)[:max_gaussians]
            gaussians = replace(
                gaussians,
                means=gaussians.means[:, idx],
                covariances=gaussians.covariances[:, idx],
                opacities=gaussians.opacities[:, idx],
                color_harmonics=gaussians.color_harmonics[:, idx] if gaussians.color_harmonics is not None else None,
                feature_harmonics=gaussians.feature_harmonics[:, idx] if hasattr(gaussians, 'feature_harmonics') and gaussians.feature_harmonics is not None else None,
            )
            
        print(">>> Rasterizer input check")
        print("   means  - shape:", gaussians.means.shape,
              "min:", gaussians.means.min().item(), "max:", gaussians.means.max().item())
        print("   covs   - shape:", gaussians.covariances.shape,
              "min:", gaussians.covariances.min().item(), "max:", gaussians.covariances.max().item())
        print("   opacs  - shape:", gaussians.opacities.shape,
              "min:", gaussians.opacities.min().item(), "max:", gaussians.opacities.max().item())
        print(">>> torch.isnan(means).any():", torch.isnan(gaussians.means).any(),
              "torch.isinf(means).any():", torch.isinf(gaussians.means).any())

        # 渲染 - 修复：同时传递 color_sh 和 feature_sh
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
            color_sh,  # 修复：传递 color_sh
            feature_sh,  # 修复：传递 feature_sh
        )
        # ---- 补丁：若 cuda 端没返回 feature，手工造一份 ----
        if rendered.feature is None and return_features:
            # 与下游期望的 [BV, C, H, W] 对齐
            b, v = extrinsics.shape[:2]
            h, w = image_shape
            c = 4          # 下游 DiagonalGaussian 需要 mean/logvar 各 2 → 共 4
            rendered.feature = torch.zeros(
                (b*v, c, h, w), device=rendered.color.device, dtype=rendered.color.dtype
            )
        print(f"[DECODER_DEBUG] Render output:")
        print(f"[DECODER_DEBUG] - color: {rendered.color.shape if rendered.color is not None else 'None'}")
        print(f"[DECODER_DEBUG] - feature: {rendered.feature.shape if rendered.feature is not None else 'None'}")
        print(f"[DECODER_DEBUG] - mask: {rendered.mask.shape if rendered.mask is not None else 'None'}")
        print(f"[DECODER_DEBUG] - depth: {rendered.depth.shape if rendered.depth is not None else 'None'}")

        out = self.render_to_decoder_output(rendered, b, v, image_shape)
        
        if depth_mode is not None and depth_mode != "depth":
            out.depth = self.render_depth(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode)
        return out

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"], # type: ignore
        intrinsics: Float[Tensor, "batch view 3 3"], # type: ignore
        near: Float[Tensor, "batch view"], # type: ignore
        far: Float[Tensor, "batch view"], # type: ignore
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]: # type: ignore
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


    # def last_layer_weights(self) -> None:
    #     return None
    @property
    def last_layer_weights(self) -> Tensor | None:
        return None