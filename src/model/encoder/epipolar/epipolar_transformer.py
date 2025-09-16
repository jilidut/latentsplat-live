from dataclasses import dataclass
from functools import partial
from typing import Optional
from src.model import get_patches

from einops import rearrange
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
        b, v, c, h, w = features.shape          # 统一 5 维

        # ===== 两行兼容代码：只处理广播，训练无副作用 =====
        for t in (near, far):                       # t 形状可能是 [B,V] 或 [B,1]
            if t.dim() == 2 and t.shape[1] == 1:    # 真正缺失 view 维
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
            # # 0. 下采样
            # print(f"[DEBUG] before downscaler: features.shape={features.shape}")
            # if self.downscaler is not None:
            #     features = rearrange(features, "b v c h w -> (b v) c h w")
            #     features = self.downscaler(features)
            #     features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
            # features_down = features
            # print(f"[DEBUG] after downscaler:  features_down.shape={features_down.shape}")

            # B, V, C, H, W = features_down.shape
            # rays_uv = sampling.xy_sample                 # [B, V, R, H', W', 2]
            # R = rays_uv.shape[2]
            # S = rays_uv.shape[3] * rays_uv.shape[4]      # 12100

            # # 1. 映射坐标
            # B, V, C, H, W = features_down.shape
            # rh = (rays_uv[..., 1] * H).long().clamp(0, H - 1)
            # rw = (rays_uv[..., 0] * W).long().clamp(0, W - 1)
            # flat_hw = (rh * W + rw).long()               # [B, V, R, H', W']

            # # 2. 按视角拆表 → 列表，每项 [B*H*W, C]  实际 [1, 128]
            # features_v = [features_down[:, v].permute(1, 2, 3, 0).reshape(H * W, C)
            #             for v in range(V)]               # V=2

            # # 3. 逐视角索引 → 每项 [B*R*S, C]  12100
            # kv_list = []
            # for v in range(V):
            #     idx_v = flat_hw[:, v].reshape(-1)        # [B*R*S]  12100
            #     kv_list.append(features_v[v][idx_v])     # [12100, 128]  ← 直接切片即可
            # kv = torch.cat(kv_list, dim=0).unsqueeze(1)  # [24200, 1, 128]

            # # 4. q 摊平
            # q_flat = rearrange(q, "b v ov r s c -> (b v ov r s) () c")  # [24200, 1, 128]

            # # 5. 过 transformer
            # out = self.transformer(kv, q_flat, b=B, v=V, h=-1, w=-1)

            # # 6. 还原
            # features = out.view(B, V, R, S, C)[..., 0]  

            # 0. 下采样（与训练分支保持一致）
            if self.downscaler is not None:
                features = rearrange(features, "b v c h w -> (b v) c h w")
                features = self.downscaler(features)
                features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
            features_down = features

            # 1. 坐标映射与索引（从采样器拿到 uv 样本）
            B, V, C, H, W = features_down.shape
            rays_uv = sampling.xy_sample                 # [B, V, R, H', W', 2]
            R = rays_uv.shape[2]
            S = rays_uv.shape[3] * rays_uv.shape[4]

            rh = (rays_uv[..., 1] * H).long().clamp(0, H - 1)
            rw = (rays_uv[..., 0] * W).long().clamp(0, W - 1)
            flat_hw = (rh * W + rw).long()               # [B, V, R, H', W']

            # 2. 按视角拆表 → 列表，每项 [H*W, C]
            features_v = [
                features_down[:, vi].permute(1, 2, 3, 0).reshape(H * W, C)
                for vi in range(V)
            ]

            # 3. 逐视角索引 → 每项 [B*R*S, C]，然后拼接成 kv
            kv_list = []
            for vi in range(V):
                idx_v = flat_hw[:, vi].reshape(-1)        # [B*R*S]
                kv_list.append(features_v[vi][idx_v])     # [B*R*S, C]
            kv = torch.cat(kv_list, dim=0).unsqueeze(1)   # [B*V*R*S, 1, C]  （注意：顺序为各视角块依次拼接）

            # 4. 准备 q：从 sampling.features 中抽取射线并做 patch（get_patches 返回 (q_patches, coords)）
            # q 原形状 (b, v, ov, r, s, c) 或类似，先把 ov 合并
            q = rearrange(q, "b v ov r s c -> b v ov r s c")  # 确保形状一致
            q = rearrange(q, "b v ov r s c -> (b v ov) r s c")  # (BVO) x R x S x C
            q, patch_coord = get_patches(q, self.cfg.test_num_rays, stride=1)  # q -> (BVO_selected) x R_sel x S x C
            # 这里 get_patches 应返回与 cfg.test_num_rays 对应的 R_sel（通常是一个平方数）
            # 恢复到 (b, v, r, s, c) 以便后续 reshape
            # 我们需要保留最外层真实 batch b 与 v，下面用 reshape 恢复
            # 先计算真实 b 和 v（保持原输入大小）
            # 注意：patch 输出可能改变第一个维度的组合形式，这里假设 get_patches 保持 (b*v*ov, ...)
            # 若 get_patches 的返回格式不同，请据其实际返回调整这一段。
            q = rearrange(q, "(bvo) r s c -> bvo r s c", bvo=(b * v * (q.shape[0] // (b * v))), r=q.shape[1], s=q.shape[2], c=q.shape[3])
            # 将 bvo 拆回 b, v, ov（这里 ov=1 或原始 ov），简化处理以便 reshape 回 (b, v, r, s, c)
            # 如果 get_patches 保持 (b*v, r, s, c) 则下面两行会自然工作
            try:
                q = rearrange(q, "(b v ov) r s c -> b v ov r s c", b=b, v=v)
            except Exception:
                # 若不能按 ov 恢复，尝试把 ov 置为 1
                q = rearrange(q, "(b v) r s c -> b v 1 r s c", b=b, v=v)

            # 5. 将 q 展平为 transformer 可接受的形状 (batch_seq, token, C)
            # 我们需要和 kv 对齐：kv 的 seq 长为 B*V*R*S（按上面拼接方式）
            # 将 q 先成形为 [b, v, r, s, c]，再展平为 (b*v*r, s, c)，最后变为 (b*v*r, s, c) -> (b*v*r, s, c) 以传入 transformer
            q_for_transformer = rearrange(q, "b v ov r s c -> (b v r) s c", b=b, v=v, r=R)

            # 6. 过 transformer（只调用一次）
            # 注意：kv 长度为 (B*V*R*S)，而 q_for_transformer 每条序列的 token 长为 S
            out = self.transformer(kv, q_for_transformer, b=b, v=v, h=-1, w=-1)

            # 7. 还原为 (b, v, r, s, c) 然后取 token 维度的中心或第0维（如原逻辑）
            # out 形状通常为 (b*v*r, s, c) 或 (b*v*r, 1, c) 取决于 transformer 实现，这里尽量兼容
            try:
                out_reshaped = out.view(b, v, R, S, C)
                features = out_reshaped[..., 0]  # [b, v, r, c]
            except Exception:
                # 如果 out 的 shape 是 (b*v*r, 1, c)，先 reshape到 (b, v, r, 1, c) 再取
                features = out.view(b, v, R, -1)[..., 0]  # [b, v, r, c]

            # 8. 将 test 下得到的 features 展平成 transformer 通用后续处理所期望的形状
            features = rearrange(features, "b v r c -> (b v r) () c")

        # 7. transformer 返回 (B*V*H*W, 1, C) 或 (B*V*R, 1, C)
        features = self.transformer(
            kv,
            rearrange(q, "b v ov r s c -> (b v ov r) s c"),
            b=b, v=v, h=-1, w=-1,
        )

        # ------ 动态推算 patch 网格 ------
        L = features.size(0) // (b * v)
        h_patches = int(math.sqrt(L))
        w_patches = L // h_patches
        assert h_patches * w_patches == L, f"cannot factor {L} into 2 ints"

        # 8. 重新摆回 5 维
        if self.training:
            features = rearrange(
                features, "(b v h w) () c -> b v c h w",
                b=b, v=v, h=h_patches, w=w_patches
            )
        else:
            r = q.shape[2]
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
        x = x.squeeze(1)          # -> (T, C)
        x = self.mlp(x) + x       # 残差
        return x.unsqueeze(1)     # -> (T, 1, C)
        
