from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Shaped
from torch import Tensor, nn

from ....geometry.epipolar_lines import project_rays
from ....geometry.projection import get_world_rays, sample_image_grid
from ....misc.heterogeneous_pairings import (
    Index,
    generate_heterogeneous_index,
    generate_heterogeneous_index_transpose,
)


@dataclass
class EpipolarSampling:
    features: Float[Tensor, "batch view other_view ray sample channel"]
    valid: Bool[Tensor, "batch view other_view ray"]
    xy_ray: Float[Tensor, "batch view ray 2"]
    xy_sample: Float[Tensor, "batch view other_view ray sample 2"]
    xy_sample_near: Float[Tensor, "batch view other_view ray sample 2"]
    xy_sample_far: Float[Tensor, "batch view other_view ray sample 2"]
    origins: Float[Tensor, "batch view ray 3"]
    directions: Float[Tensor, "batch view ray 3"]
    depths: Float[Tensor, "batch view other_view ray sample"] 


class EpipolarSampler(nn.Module):
    num_samples: int
    index_v: Index
    transpose_v: Index
    transpose_ov: Index

    def __init__(
        self,
        num_views: int,
        num_samples: int,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

        # Generate indices needed to sample only other views.
        _, index_v = generate_heterogeneous_index(num_views)
        t_v, t_ov = generate_heterogeneous_index_transpose(num_views)
        self.register_buffer("index_v", index_v, persistent=False)
        self.register_buffer("transpose_v", t_v, persistent=False)
        self.register_buffer("transpose_ov", t_ov, persistent=False)

    def forward(
        self,
        images: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
    ) -> EpipolarSampling:
        device = images.device
        b, v, _, _, _ = images.shape

        # Generate the rays that are projected onto other views.
        xy_ray, origins, directions = self.generate_image_rays(
            images, extrinsics, intrinsics
        )

        # Select the camera extrinsics and intrinsics to project onto. For each context
        # view, this means all other context views in the batch.
        projection = project_rays(
            rearrange(origins, "b v r xyz -> b v () r xyz"),
            rearrange(directions, "b v r xyz -> b v () r xyz"),
            rearrange(self.collect(extrinsics), "b v ov i j -> b v ov () i j"),
            rearrange(self.collect(intrinsics), "b v ov i j -> b v ov () i j"),
            rearrange(near, "b v -> b v () ()"),
            rearrange(far, "b v -> b v () ()"),
        )

        # Generate sample points.
        s = self.num_samples
        sample_depth = (torch.arange(s, device=device) + 0.5) / s
        # sample_depth = rearrange(sample_depth, "s -> s ()")
        xy_min = projection["xy_min"].nan_to_num(posinf=0, neginf=0) 
        xy_min = xy_min * projection["overlaps_image"][..., None]
        xy_min = rearrange(xy_min, "b v ov r xy -> b v ov r () xy")
        xy_max = projection["xy_max"].nan_to_num(posinf=0, neginf=0) 
        xy_max = xy_max * projection["overlaps_image"][..., None]
        xy_max = rearrange(xy_max, "b v ov r xy -> b v ov r () xy")
        sample_depth = sample_depth.view(1, 1, 1, 1, s, 1)   
        xy_sample = xy_min + sample_depth * (xy_max - xy_min)

                # ---------- ① 补齐射线数到整张图，保证后面 reshape 不崩 ----------
        _, _, _, r_all = projection["overlaps_image"].shape   # 当前有效射线数
        h, w = images.shape[-2:]                               # 原图高宽
        r_needed = h * w                                       # 目标射线数 = 整张图
        if r_all != r_needed:
            # 1. 把 mask 补成 True
            flat_mask = projection["overlaps_image"].reshape(-1)          # (B*V*OV*R,)
            n_pad = r_needed - r_all
            # 随机挑 n_pad 条已有射线做复制（避免全 0）
            idx_true = torch.where(flat_mask)[0]
            pad_idx = idx_true[torch.randint(len(idx_true), (n_pad,), device=idx_true.device)]
            flat_mask = torch.cat([flat_mask, flat_mask[pad_idx]])        # 补 True
            projection["overlaps_image"] = flat_mask.reshape(b, v, v-1, r_needed)

            # 2. 把 xy_sample 补到同样长度
            flat_xy = xy_sample.reshape(-1, s, 2)                         # (N, 484, 2)
            pad_xy = flat_xy[pad_idx]                                     # 复制同样条数
            flat_xy = torch.cat([flat_xy, pad_xy], 0)                     # (N+n_pad, 484, 2)
            xy_sample = flat_xy.reshape(b, v, v-1, r_needed, s, 2)       # 现在长度对齐了
        # ----------------------------------------------------------

        # ---------- ② 无效射线对应的样本直接置 0 ----------
        xy_sample = xy_sample * projection["overlaps_image"][..., None, None]

        # 2. 提前保护：没有有效样本直接返回空张量
        if projection["overlaps_image"].sum() == 0:
            b, v, c, h, w = images.shape
            r = h * w                                      # 射线数 = 整张图
            s = self.num_samples
            return EpipolarSampling(
                features=images.new_zeros(b, v, v-1, r, s, c),
                valid=projection["overlaps_image"],          # 全 False
                xy_ray=xy_ray,
                xy_sample=xy_sample.new_zeros(b, v, v-1, r, s, 2),
                xy_sample_near=xy_sample.new_zeros(b, v, v-1, r, s, 2),
                xy_sample_far=xy_sample.new_zeros(b, v, v-1, r, s, 2),
                origins=origins,
                directions=directions,
            )

        samples = self.transpose(xy_sample)
        samples = F.grid_sample(
            rearrange(images, "b v c h w -> (b v) c h w"),
            rearrange(2 * samples - 1, "b v ov r s xy -> (b v) (ov r s) () xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


        samples = rearrange(
            samples, "(b v) c (ov r s) () -> b v ov r s c", b=b, v=v, ov=v - 1, s=s
        )
        samples = self.transpose(samples)          # (B, V, OV, R, S, C)
        # ---------- 构造 depths ----------
        b, v, ov, r, s, c = samples.shape
        depths = sample_depth.view(1, 1, 1, 1, s, 1).expand(b, v, ov, r, -1, 1).squeeze(-1)


        print("[EpipolarSampler] samples.shape:", samples.shape)
        print("[EpipolarSampler] r_needed:", h * w)
        print("[EpipolarSampler] r_all after pad:", samples.shape[3])

        # Zero out invalid samples.
        samples = samples * projection["overlaps_image"][..., None, None]

        half_span = 0.5 / s
        return EpipolarSampling(
            features=samples,
            valid=projection["overlaps_image"],
            xy_ray=xy_ray,
            xy_sample=xy_sample,
            xy_sample_near=xy_min + (sample_depth - half_span) * (xy_max - xy_min),
            xy_sample_far=xy_min + (sample_depth + half_span) * (xy_max - xy_min),
            origins=origins,
            directions=directions,
            depths=depths,   
        )

    def generate_image_rays(
        self,
        images: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
    ) -> tuple[
        Float[Tensor, "batch view ray 2"],  # xy
        Float[Tensor, "batch view ray 3"],  # origins
        Float[Tensor, "batch view ray 3"],  # directions
    ]:
        """Generate the rays along which Gaussians are defined. For now, these rays are
        simply arranged in a grid.
        """
        b, v, _, h, w = images.shape
        xy, _ = sample_image_grid((h, w), device=images.device)
        origins, directions = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            rearrange(extrinsics, "b v i j -> b v () i j"),
            rearrange(intrinsics, "b v i j -> b v () i j"),
        )
        return repeat(xy, "h w xy -> b v (h w) xy", b=b, v=v), origins, directions

    def transpose(
        self,
        x: Shaped[Tensor, "batch view other_view *rest"],
    ) -> Shaped[Tensor, "batch view other_view *rest"]:
        b, v, ov, *_ = x.shape
        t_b = torch.arange(b, device=x.device)
        t_b = repeat(t_b, "b -> b v ov", v=v, ov=ov)
        t_v = repeat(self.transpose_v, "v ov -> b v ov", b=b)
        t_ov = repeat(self.transpose_ov, "v ov -> b v ov", b=b)
        return x[t_b, t_v, t_ov]

    def collect(
        self,
        target: Shaped[Tensor, "batch view ..."],
    ) -> Shaped[Tensor, "batch view view-1 ..."]:
        b, v, *_ = target.shape
        index_b = torch.arange(b, device=target.device)
        index_b = repeat(index_b, "b -> b v ov", v=v, ov=v - 1)
        index_v = repeat(self.index_v, "v ov -> b v ov", b=b)
        return target[index_b, index_v]
