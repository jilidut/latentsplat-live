from dataclasses import dataclass
from functools import partial
from typing import Optional

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.epipolar_lines import get_depth
from ....global_cfg import get_cfg
from ...encodings.positional_encoding import PositionalEncoding
from ...transformer.transformer import Transformer
from .conversions import depth_to_relative_disparity
from .epipolar_sampler import EpipolarSampler, EpipolarSampling
from .image_self_attention import ImageSelfAttention, ImageSelfAttentionCfg
import math  

@dataclass
class EpipolarTransformerCfg:
    self_attention: ImageSelfAttentionCfg
    num_octaves: int
    num_layers: int
    num_heads: int
    # num_samples: int
    d_dot: int
    d_mlp: int
    downscale: int
    num_samples: int = 484
    num_context_views: int = 2  
    test_num_rays: int = 512              
    test_num_points_per_ray: int = 484 


class EpipolarTransformer(nn.Module):
    cfg: EpipolarTransformerCfg
    epipolar_sampler: EpipolarSampler
    depth_encoding: nn.Sequential
    transformer: Transformer
    downscaler: Optional[nn.Conv2d]
    upscaler: Optional[nn.ConvTranspose2d]
    upscale_refinement: Optional[nn.Sequential]

    def __init__(
        self,
        cfg: EpipolarTransformerCfg,
        d_in: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.epipolar_sampler = EpipolarSampler(
            # get_cfg().dataset.view_sampler.num_context_views,
            cfg.num_context_views,   # 假设 cfg 里已经有这个字段
            cfg.num_samples,
        )
        if self.cfg.num_octaves > 0:
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(cfg.num_octaves)),
                nn.Linear(pe.d_out(1), d_in),
            )
        feed_forward_layer = partial(ConvFeedForward, cfg.self_attention)
        self.transformer = Transformer(
            d_in,
            cfg.num_layers,
            cfg.num_heads,
            cfg.d_dot,
            cfg.d_mlp,
            selfatt=False,
            kv_dim=d_in,
            feed_forward_layer=feed_forward_layer,
        )

        if cfg.downscale > 1:
            self.downscaler = nn.Conv2d(d_in, d_in, cfg.downscale, cfg.downscale)
            # self.upscaler = nn.ConvTranspose2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscaler = nn.ConvTranspose2d(d_in, d_in, 2, 2)
            self.upscale_refinement = nn.Sequential(
                nn.Conv2d(d_in, d_in * 2, 7, 1, 3),
                nn.GELU(),
                nn.Conv2d(d_in * 2, d_in, 7, 1, 3),
            )
        else:
            self.downscaler = None
            self.upscaler = None
            self.upscale_refinement = None

    def forward(
        self,
        features: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
    ) -> tuple[Float[Tensor, "batch view channel height width"], EpipolarSampling]:
        b, v, c, h, w = features.shape          # 统一 5 维

        # 1. 可选下采样
        if self.downscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.downscaler(features)
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        # 2. 采样
        sampling = self.epipolar_sampler(
            features, extrinsics, intrinsics, near, far
        )

        # 3. 早停：无有效样本
        if sampling.features.shape[2] == 0:          # other_view 维
            return features.new_zeros(b, v, c, h, w), sampling

        # 4. 训练/测试分叉
        if self.training:
            q = sampling.features                           # (B,V,OV,H,W,C)
        else:
            q = sampling.features[:, :, : self.cfg.test_num_rays]  # (B,V,R,H,W,C)

        # 5. 可选深度编码
        if self.cfg.num_octaves > 0:
            collect = self.epipolar_sampler.collect
            depths = get_depth(
                rearrange(sampling.origins, "b v r xyz -> b v () r () xyz"),
                rearrange(sampling.directions, "b v r xyz -> b v () r () xyz"),
                sampling.xy_sample,
                rearrange(collect(extrinsics), "b v ov i j -> b v ov () () i j"),
                rearrange(collect(intrinsics), "b v ov i j -> b v ov () () i j"),
            )
            depths = depths.clip(
                near[..., None, None, None],
                far[..., None, None, None],
            )
            depths = depth_to_relative_disparity(
                depths,
                rearrange(near, "b v -> b v () () ()"),
                rearrange(far, "b v -> b v () () ()"),
            )
            q = q + self.depth_encoding(depths[..., None])
        else:
            q = q

        # 6. 构造 kv
        if self.training:
            kv = rearrange(features, "b v c h w -> (b v h w) () c")
        else:
            rays_uv = sampling.xy_sample[:, :, : self.cfg.test_num_rays]  # (B,V,R,2)
            rh = (rays_uv[..., 1] * (h // self.cfg.downscale)).long().clamp(0, h // self.cfg.downscale - 1)
            rw = (rays_uv[..., 0] * (w // self.cfg.downscale)).long().clamp(0, w // self.cfg.downscale - 1)
            features_down = features
            if self.downscaler is not None:
                features_down = rearrange(features_down, "b v c h w -> (b v) c h w")
                features_down = self.downscaler(features_down)
                features_down = rearrange(features_down, "(b v) c h w -> b v c h w", b=b, v=v)
            kv = features_down[
                torch.arange(b)[:, None, None],
                torch.arange(v)[None, :, None],
                :,
                rh,
                rw,
            ]  # (B,V,R,C)
            kv = rearrange(kv, "b v r c -> (b v r) () c")

        # 7. transformer
        features = self.transformer(
            kv,
            rearrange(q, "b v ov r s c -> (b v ov r) s c"),
            b=b,
            v=v,
            h=-1,
            w=-1,
        )  # 返回 (B*V*H*W,1,C) 或 (B*V*R,1,C)

        print(f"[Transformer] features.shape={features.shape}, expected B*V*H*W={b*v*h*w}")
        
        # 8. 重新摆回 5 维
        if self.training:
            features = rearrange(features, "(b v h w) () c -> b v c h w", b=b, v=v, h=h, w=w)
        else:
            r = q.shape[2]          # 测试阶段采样数
            features = rearrange(features, "(b v r) () c -> b v r c", b=b, v=v, r=r)

        # 9. 可选上采样（仅训练）
        if self.upscaler is not None and self.training:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.upscaler(features)
            features = self.upscale_refinement(features) + features
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        return features, sampling


class ConvFeedForward(nn.Module):
    def __init__(
        self,
        self_attention_cfg: ImageSelfAttentionCfg,
        d_in: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_in),
            nn.Dropout(dropout),
        )
        self.self_attention = nn.Identity()   # 先占位
        

    def forward(
        self,
        x: Float[Tensor, "batch token dim"],
        b: int,
        v: int,
        h: int,
        w: int,
    ) -> Float[Tensor, "batch token dim"]:
        # x = rearrange(x, "(b v h w) () c -> (b v) c h w", b=b, v=v, h=h, w=w)
        # # ---- 打印输入 shape ----
        # print(f"[before SA] x.shape = {x.shape},  h={h}, w={w}")
        # x = self.layers(self.self_attention(x) + x)
        # # ---- 打印输出 shape ----
        # print(f"[after  SA] x.shape = {x.shape}")
        # return rearrange(x, "(b v) c h w -> (b v h w) () c", b=b, v=v, h=h, w=w)
         # x: (B*V*R, 1, C) 或 (B*V*H*W, 1, C)
        x = x.squeeze(1)          # -> (T, C)
        x = self.mlp(x) + x       # 残差
        return x.unsqueeze(1)     # -> (T, 1, C)
        
