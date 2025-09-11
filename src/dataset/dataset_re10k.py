# dataset_re10k.py  (2025-06-25 修订版)
import json
import random
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler, ViewSamplerEvaluation


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.force_shuffle = force_shuffle

        # ---- 收集 chunk ----
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(p for p in root.iterdir() if p.suffix == ".torch")
            self.chunks.extend(root_chunks)

        if cfg.overfit_to_scene is not None:
            chunk_path = self.index[cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

        print(f"[DEBUG] stage={self.stage}, data_stage={self.data_stage}")
        print(f"[DEBUG] chunks={len(self.chunks)}, index={len(self.index)}")

    # ---------------- 迭代器 ----------------
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # 1. 先按 worker 分片（train/val 也做，防止重复加载）
        if worker_info is not None:
            per_worker = len(self.chunks) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = None if worker_id == worker_info.num_workers - 1 else (worker_id + 1) * per_worker
            chunks = self.chunks[start:end]
        else:
            chunks = self.chunks

        # 2. 打乱
        if self.stage in ("train", "val") or self.force_shuffle:
            chunks = random.sample(chunks, k=len(chunks))

        for chunk_path in chunks:
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in ("train", "val"):
                chunk = random.sample(chunk, k=len(chunk))

            for example in chunk:
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]
                num_views = extrinsics.shape[0]

                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                try:
                    view_indices = self.view_sampler.sample(scene, num_views)
                except ValueError:
                    continue

                for view_index in view_indices:
                    ctx_idx, tgt_idx = view_index.context, view_index.target

                    # ---- 加载图像 ----
                    ctx_imgs = [example["images"][i.item()] for i in ctx_idx]
                    tgt_imgs = [example["images"][i.item()] for i in tgt_idx]
                    ctx_imgs = self._convert_images(ctx_imgs)
                    tgt_imgs = self._convert_images(tgt_imgs)

                    if ctx_imgs.shape[1:] != (3, 360, 640) or tgt_imgs.shape[1:] != (3, 360, 640):
                        print(f"Skip bad shape {scene}  ctx={ctx_imgs.shape}  tgt={tgt_imgs.shape}")
                        continue

                    # ----  baseline=1 归一化 ----
                    ctx_extr = extrinsics[ctx_idx]
                    scale = 1.0
                    if ctx_extr.shape[0] == 2 and self.cfg.make_baseline_1:
                        a, b = ctx_extr[:, :3, 3]
                        baseline = (a - b).norm()
                        if baseline < self.cfg.baseline_epsilon:
                            print(f"Skip {scene}  baseline={baseline:.4f}")
                            continue
                        extrinsics[:, :3, 3] /= baseline
                        scale = baseline

                    sample = {
                        "context": {
                            "extrinsics": extrinsics[ctx_idx],
                            "intrinsics": intrinsics[ctx_idx],
                            "image": ctx_imgs,
                            "near": self.get_bound("near", len(ctx_idx)) / scale,
                            "far": self.get_bound("far", len(ctx_idx)) / scale,
                            "index": ctx_idx,
                        },
                        "target": {
                            "extrinsics": extrinsics[tgt_idx],
                            "intrinsics": intrinsics[tgt_idx],
                            "image": tgt_imgs,
                            "near": self.get_bound("near", len(tgt_idx)) / scale,
                            "far": self.get_bound("far", len(tgt_idx)) / scale,
                            "index": tgt_idx,
                        },
                        "scene": scene,
                    }
                    if self.stage == "train" and self.cfg.augment:
                        sample = apply_augmentation_shim(sample)
                    yield apply_crop_shim(sample, tuple(self.cfg.image_shape))

    # ---------------- 辅助 ----------------
    def convert_poses(
        self, poses: Float[Tensor, "batch 18"]
    ) -> tuple[Float[Tensor, "batch 4 4"], Float[Tensor, "batch 3 3"]]:
        b, _ = poses.shape
        intr = torch.eye(3, dtype=torch.float32).repeat(b, 1, 1)
        fx, fy, cx, cy = poses[:, :4].T
        intr[:, 0, 0] = fx
        intr[:, 1, 1] = fy
        intr[:, 0, 2] = cx
        intr[:, 1, 2] = cy

        w2c = torch.eye(4, dtype=torch.float32).repeat(b, 1, 1)
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intr

    def _convert_images(
        self, images: list[UInt8[Tensor, "..."]]
    ) -> Float[Tensor, "batch 3 height width"]:
        return torch.stack([self.to_tensor(Image.open(BytesIO(img.numpy().tobytes()))) for img in images])

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return value.repeat(num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        return "test" if self.stage == "val" else self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged = {}
        stages = ("test", "train") if self.cfg.overfit_to_scene else (self.data_stage,)
        for stage in stages:
            for root in self.cfg.roots:
                idx_file = root / stage / "index.json"
                if not idx_file.exists():          # 🔥 必改：跳过不存在
                    continue
                with idx_file.open("r") as f:
                    sub = {k: Path(root / stage / v) for k, v in json.load(f).items()}
                assert not (merged.keys() & sub.keys()), "duplicate scene key"
                merged.update(sub)
        return merged

    def __len__(self) -> int:
        return self.view_sampler.total_samples if isinstance(self.view_sampler, ViewSamplerEvaluation) else len(self.index)