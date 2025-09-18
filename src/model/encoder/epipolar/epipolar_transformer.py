# src/model/encoder/epipolar/epipolar_transformer.py
from dataclasses import dataclass
from functools import partial
from typing import Optional


from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
import torch

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
        b, v, c, h, w = features.shape

        # ===== 两行兼容代码：只处理广播，训练无副作用 =====
        for t in (near, far):  # t 形状可能是 [B,V] 或 [B,1]
            if t.dim() == 2 and t.shape[1] == 1:  # 真正缺失 view 维
                t.data = t.expand(b, v)

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
        if sampling.features.shape[2] == 0:  # other_view 维
            return features.new_zeros(b, v, c, h, w), sampling

        # 4. 训练/测试分叉
        if self.training:
            q = sampling.features  # (B,V,OV,H,W,C)
        else:
            q = sampling.features[:, :, :, :self.cfg.test_num_rays]  # (B,V,OV,R,H,W,C)
            # 1. 压维 → (B,V,OV,R,S,C)  S=H*W
            if q.ndim == 7:
                b_, v_, ov_, r_, h_, w_, c_ = q.shape
                q = q.view(b_, v_, ov_, r_, h_*w_, c_)
            else:  # 6 维
                b_, v_, ov_, r_, s_, c_ = q.shape

            # 2. 关键：测试时若 S=1，先 pad 到 2，防止后面 squeeze 掉
            if q.size(-2) == 1 and not self.training:
                q = torch.cat([q, q], dim=-2)

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

        # 6. 构造 kv：整幅图展平
        kv = rearrange(features, "b v c h w -> (b v h w) () c")  # (B*V*H*W, 1, C)

        # 7. 统一 reshape q
        q_flat = rearrange(q, "b v ov r s c -> (b v ov r) s c")  # (B*V*OV*R, S, C)

        # 8. 过 Transformer
        features = self.transformer(kv, q_flat, b=b, v=v, h=-1, w=-1)  # (T, S, C)

        # 9. 摆回多维 – 训练/测试分叉
        if self.training:
            L = features.size(0) // (b * v)
            h_patches = int(math.sqrt(L))
            w_patches = L // h_patches
            assert h_patches * w_patches == L, f"cannot factor {L} into 2 ints"
            features = rearrange(
                features, "(b v h w) () c -> b v c h w",
                b=b, v=v, h=h_patches, w=w_patches
            )
        else:
            # 9. 摆回 6 维 – 测试专用
            features = rearrange(features, "(b v ov r) s c -> b v ov r s c",
                                 b=b, v=v, ov=q.shape[2], r=q.shape[3])
            features = rearrange(features, "b v ov r s c -> b v c (ov r) s")

        # 10. 可选上采样（仅训练）
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
        if not self.training and x.size(1) == 1:   # 仅测试 & token=1
            x = x.expand(-1, 2, -1)                # pad 到 S=2
        if self.training:
            x = x.squeeze(1)                       # 训练正常走 (T,C)
            x = self.mlp(x) + x
            return x.unsqueeze(1)                  # (T,1,C)
        else:
            T, S, C = x.shape
            x = x.reshape(-1, C)        # (T*S, C)
            x = self.mlp(x) + x
            x = x.reshape(T, S, C)         # (T,S,C)
            return x 
       
        
