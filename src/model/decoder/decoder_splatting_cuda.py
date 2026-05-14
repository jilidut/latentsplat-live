# src/model/decoder/decoder_splatting_cuda.py
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from ..diagonal_gaussian_distribution import DiagonalGaussianDistribution
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, RenderOutput, render_depth_cuda
from .decoder import Decoder, DecoderOutput
from torch import nn
from typing import Optional 
import torch.nn.functional as F
from ..types import VariationalGaussians   # 新增
# from loss.geom_sem_contrast import GeomSemContrastLoss       
# from loss.unc_sem_couple   import UncSemCoupleLoss 
@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    enable_residual: bool = True 


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        background_color: list[float] = [0., 0., 0.],
        variational: bool = False,
    ) -> None:
        super().__init__(cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        self.variational = variational
        # 新增
        self.feature_out_ch = getattr(cfg, 'feature_out_ch', 4) 
        self.res_w = nn.Parameter(torch.tensor(0.01))
        self.sem2rgb = nn.Conv1d(32, 3, kernel_size=1)
        nn.init.zeros_(self.sem2rgb.weight)
        nn.init.zeros_(self.sem2rgb.bias)
        self.enable_residual = getattr(cfg, 'enable_residual', True)  # 控制残差
        # 新增

    def render_to_decoder_output(
        self,
        render_output: RenderOutput,
        b: int,
        v: int
    ) -> DecoderOutput:
        # 新增
        if render_output.feature is not None:
            c = render_output.feature.shape[1]
            if c != self.feature_out_ch:
                import warnings, torch.nn.functional as F
                warnings.warn(f"[Decoder] feature channel {c} ≠ config {self.feature_out_ch}；已自动截断/补零到 {self.feature_out_ch}")
                if c > self.feature_out_ch:
                    render_output.feature = render_output.feature[:, :self.feature_out_ch]
                else:
                    render_output.feature = F.pad(render_output.feature, (0, 0, 0, 0, 0, self.feature_out_ch - c))
        # ========================================
        feature_posterior = None
        features = None 
        if render_output.feature is not None:
            features = rearrange(render_output.feature, "(b v) c h w -> b v c h w", b=b, v=v)
            mean_flat = rearrange(features, "b v c h w -> b (v h w) c")
            if self.variational:
                # 真实 logvar 应该从网络来，这里先临时用 0 表示“确信”
                logvar_flat = torch.zeros_like(mean_flat)
                feature_posterior = DiagonalGaussianDistribution(
                    torch.cat([mean_flat, logvar_flat], dim=-1), dim=-1
                )
            else:
                # 无论是否 variational，都给一个确定性 posterior
                mean_flat   = rearrange(features, "b v c h w -> b (v h w) c")
                logvar_flat = torch.full_like(mean_flat, -20.)          # 几乎无方差
                feature_posterior = DiagonalGaussianDistribution(
                    torch.cat([mean_flat, logvar_flat], dim=-1), dim=-1
                )

        #原代码
        # if render_output.feature is not None:
        #     features = rearrange(render_output.feature, "(b v) c h w -> b v c h w", b=b, v=v)
        #     # NOTE background feature = 0 = mean = logvar (of normal distribution)
        #     mean, logvar = features.chunk(2, dim=2) if self.variational \
        #         else (features, (1-rearrange(render_output.mask.detach(), "(b v) h w -> b v () h w", b=b, v=v)).log().expand_as(features))
        #     feature_posterior = DiagonalGaussianDistribution(mean, logvar)
        # else:
        #     feature_posterior = None
        return DecoderOutput(
            color=rearrange(render_output.color, "(b v) c h w -> b v c h w", b=b, v=v) if render_output.color is not None else None,
            feature_posterior=feature_posterior,
            mask=rearrange(render_output.mask, "(b v) h w -> b v h w", b=b, v=v),
            depth=rearrange(render_output.depth, "(b v) h w -> b v h w", b=b, v=v),
            feature_map=features,  #新增
        )

    def forward(
        self,
        # gaussians: Gaussians,
        gaussians: Gaussians | VariationalGaussians, # 新增
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        return_colors: bool = True,
        return_features: bool = True,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape

        #新增
        if isinstance(gaussians, VariationalGaussians):
            gaussians_flat = gaussians.sample() if self.variational in ("gaussians", "none") else gaussians.flatten()
            gaussian_latent = gaussians.gaussian_latent         
        else:  # 普通 Gaussians
            gaussians_flat = gaussians
            gaussian_latent = None
        color_sh = repeat(gaussians_flat.color_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) \
            if return_colors and gaussians_flat.color_harmonics is not None else None
        feature_sh = repeat(gaussians_flat.feature_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) \
            if return_features and gaussians_flat.feature_harmonics is not None else None

        # [关键修复] 存在且 >4 才切，保证不炸梯度
        if feature_sh is not None and feature_sh.shape[2] > 4:
            feature_sh = feature_sh[:, :, :feature_sh.shape[2]//2, :]

        print(f"[UP2] feature_sh={feature_sh.shape if feature_sh is not None else None}")
        #新增

        # color_sh = repeat(gaussians.color_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) \
        #     if return_colors and gaussians.color_harmonics is not None else None
        # feature_sh = repeat(gaussians.feature_harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) \
        #     if return_features and gaussians.feature_harmonics is not None else None
        rendered: RenderOutput = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            # repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            # repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            # repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            repeat(gaussians_flat.means,     "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians_flat.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians_flat.opacities, "b g -> (b v) g", v=v),
            color_sh,
            feature_sh
        )

        out = self.render_to_decoder_output(rendered, b, v)
        # if depth_mode is not None and depth_mode != "depth":
                #     out.depth = self.render_depth(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode)
                # return out

        # 新增改变
        # ========== decoder_splatting_cuda.py :: forward() 残差分支 ==========
        # if gaussian_latent is not None and out.color is not None
        if self.enable_residual and gaussian_latent is not None and out.color is not None:
            z = gaussian_latent.sample()                   
            v_real = out.color.shape[0] // b
            z = z[:, :v_real]       
            B_V = b * v_real
            N = z.shape[2]

            # 1. 三流特征
            sem = z[..., 32:64]
            unc = z[..., 64:65]

            # ---- 维度拿清楚 ----
            B, V, N, _ = sem.shape           # (B, V, N, 32)
            BV = B * V

            # ---- 2️⃣ 语义 → RGB 残差 ----
            sem_flat = rearrange(sem, "b v n c -> (b v) c n")     # (BV, 32, N)
            delta = self.sem2rgb(sem_flat)                        # (BV, 3, N)
            delta = rearrange(delta, "bv c n -> bv n c")          # (BV, N, 3)

            # ---- 3️⃣ 不确定性门控 ----
            gate = torch.sigmoid(unc).clamp(0.01, 0.99)           # (B, V, N, 1)
            gate = rearrange(gate, "b v n c -> (b v) n c")        # (BV, N, 1)

            delta = gate * delta                                  # (BV, N, 3)

            # ---- 4️⃣ N 维 → 像素聚合 ----
            if hasattr(rendered, "gaussian_weights") and rendered.gaussian_weights is not None:
                weights = rendered.gaussian_weights              # (BV, N)  ← 最优
            else:
                raw_opa = gaussians_flat.opacities[:, :N]        # (B, N)
                raw_opa = repeat(raw_opa, "b n -> (b v) n", v=V)  # (BV, N)
                weights = raw_opa.clamp(min=1e-6)                # 避免 0 即可，不强制和为 1

            delta_img = (weights.unsqueeze(-1) * delta).sum(dim=1)   # (BV, 3)

            # ---- 5️⃣ reshape 回 decoder 5D 结构 ----
            delta_5d = delta_img.view(B, V, 1, 1, 3)        # (B, V, 1, 1, 3)

            # ---- 6️⃣ 安全写回 ----
            out.color[..., :3] = (out.color[..., :3] + self.res_w * delta_5d).clamp(0, 1)
        return out
        # ============================================================

    def render_depth(
        self,
        # gaussians: Gaussians,
        gaussians: Gaussians | VariationalGaussians, 
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
         #  新增---------- ----------
        if isinstance(gaussians, VariationalGaussians):
            gaussians_flat = gaussians.sample() if self.variational in ("gaussians", "none") else gaussians.flatten()
        else:
            gaussians_flat = gaussians
        # 新增--------------------------------------
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians_flat.means,     "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians_flat.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians_flat.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)


    def last_layer_weights(self) -> None:
        return None


if __name__ == "__main__":
    from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
    cfg = DecoderSplattingCUDACfg(name="splatting_cuda")
    dec = DecoderSplattingCUDA(cfg)
    print("✅ Decoder 已带 GSU 残差分支，参数="
          , dec.sem2rgb.weight.shape, "weight=", dec.res_w.item())
    #python -m src.model.decoder.decoder_splatting_cuda