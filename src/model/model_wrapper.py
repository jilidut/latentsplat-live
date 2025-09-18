# src/mode/model_wrapper.py
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from math import prod
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Protocol, runtime_checkable, Tuple
from warnings import warn

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, optim
from torch.nn import Module, Parameter
from torchvision.transforms.functional import resize

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import LossGroupCfg, LossGroup, get_loss_group
from ..misc.benchmarker import Benchmarker
from ..misc.fraction_utils import get_integer, get_inv
from ..misc.image_io import prep_image, save_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from ..visualization.camera_trajectory.wobble import generate_wobble, generate_wobble_transformation
from ..visualization.color_map import apply_depth_color_map
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .autoencoder.autoencoder import Autoencoder
from .decoder.decoder import Decoder, DepthRenderingMode
from .discriminator.discriminator import Discriminator
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .types import Prediction, GroundTruth, VariationalGaussians, VariationalMode


# ------------------------------------------------------------------ #
#                          调试开关                                   #
# ------------------------------------------------------------------ #
DEBUG = False

def log(msg: str) -> None:
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ------------------------------------------------------------------ #
#                          配置结构体                                 #
# ------------------------------------------------------------------ #
@dataclass
class LRSchedulerCfg:
    name: str
    kwargs: Dict[str, Any] | None = None


@dataclass
class FreezeCfg:
    autoencoder: bool = False
    encoder: bool = False
    decoder: bool = False
    # discrimininator: bool = False
    discriminator: bool = False 


@dataclass
class GeneratorOptimizerCfg:
    name: str
    autoencoder_lr: float
    scale_autoencoder_lr: bool
    lr: float
    scale_lr: bool
    autoencoder_kwargs: Dict[str, Any] | None = None
    kwargs: Dict[str, Any] | None = None
    scheduler: LRSchedulerCfg | None = None
    gradient_clip_val: float | int | None = None
    gradient_clip_algorithm: Literal["value", "norm"] = "norm"


@dataclass
class DiscriminatorOptimizerCfg:
    name: str
    lr: float
    scale_lr: bool
    kwargs: Dict[str, Any] | None = None
    scheduler: LRSchedulerCfg | None = None
    gradient_clip_val: float | int | None = None
    gradient_clip_algorithm: Literal["value", "norm"] = "norm"


@dataclass
class OptimizerCfg:
    generator: GeneratorOptimizerCfg
    discriminator: DiscriminatorOptimizerCfg | None = None


@dataclass
class TestCfg:
    output_path: Path
    decode_tile: int = 64
    num_rays: int = 64
    num_points_per_ray: int = 64


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    step_offset: int
    video_interpolation: bool = False
    video_wobble: bool = False


# ------------------------------------------------------------------ #
#                          工具函数                                   #
# ------------------------------------------------------------------ #
def freeze(m: Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def unfreeze(m: Module) -> None:
    m.train()
    for p in m.parameters():
        p.requires_grad = True


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(self, t: Float[Tensor, " t"]) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        ...
# ------------------------------------------------------------------ #
#                      Lightning 封装                                 #
# ------------------------------------------------------------------ #
class ModelWrapper(LightningModule):
    context_loss_cfg: LossGroupCfg | None = None,
    target_autoencoder_loss_cfg: LossGroupCfg | None = None,
    target_render_latent_loss_cfg: LossGroupCfg | None = None,
    target_render_image_loss_cfg: LossGroupCfg | None = None,
    target_combined_loss_cfg: LossGroupCfg | None = None,
    logger: Optional[WandbLogger]
    autoencoder: Autoencoder
    encoder: Encoder
    encode_latents: bool
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    discriminator: Discriminator | None
    gaussian_losses: LossGroup | None
    context_losses: LossGroup
    target_autoencoder_losses: LossGroup
    target_render_latent_losses: LossGroup
    target_render_image_losses: LossGroup
    target_combined_losses: LossGroup
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    freeze_cfg: FreezeCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        cfg: Any,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        freeze_cfg: FreezeCfg,
        autoencoder: Autoencoder,
        encoder: Encoder,
        encode_latents: bool,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        supersampling_factor: int = 1,
        variational: VariationalMode = "none",
        discriminator: Discriminator | None = None,
        context_loss_cfg: LossGroupCfg | None = None,
        target_autoencoder_loss_cfg: LossGroupCfg | None = None,
        target_render_latent_loss_cfg: LossGroupCfg | None = None,
        target_combined_loss_cfg: LossGroupCfg | None = None,   # ← 补上
        target_render_image_loss_cfg: LossGroupCfg | None = None,
        step_tracker: StepTracker | None = None,
    ) -> None:

        super().__init__()
        self.cfg = cfg 
        self.automatic_optimization = False
        log("[ModelWrapper] __init__ called")
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.freeze_cfg = freeze_cfg
        self.step_tracker = step_tracker
        self.supersampling_factor = supersampling_factor
        self.variational = variational

        if target_render_image_loss_cfg is None:
            target_render_image_loss_cfg = (
                cfg.loss.target.render.image
                if (hasattr(cfg, "loss") and
                    hasattr(cfg.loss, "target") and
                    hasattr(cfg.loss.target, "render") and
                    hasattr(cfg.loss.target.render, "image"))
                else None
            )

        # --- 模型 ---
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.encode_latents = encode_latents
        self.encoder_visualizer = encoder_visualizer
        self.data_shim = get_data_shim(self.encoder)
        self.decoder = decoder
        self.discriminator = discriminator

        # --- 损失 ---
        gaussian_loss_cfg = cfg.loss.gaussian
        log(f"[ModelWrapper] gaussian_loss_cfg = {gaussian_loss_cfg}")
        if gaussian_loss_cfg is not None:
            self.gaussian_losses = get_loss_group("gaussian", gaussian_loss_cfg)
            log(f"[ModelWrapper] gaussian_losses = {self.gaussian_losses}")
        else:
            log("[ModelWrapper] gaussian_loss_cfg is None")
            self.gaussian_losses = None


        # 补上这五行
        self.context_losses = get_loss_group("context", context_loss_cfg)
        self.target_autoencoder_losses = get_loss_group("target/autoencoder", target_autoencoder_loss_cfg)
        self.target_render_latent_losses = get_loss_group("target/render/latent", target_render_latent_loss_cfg)
        self.target_render_image_losses = get_loss_group("target/render/image", target_render_image_loss_cfg)
        self.target_combined_losses = get_loss_group("target/combined", target_combined_loss_cfg)

        # --- 梯度 / 冻结 ---
        if self.freeze_cfg.autoencoder:
            freeze(self.autoencoder)
        if self.freeze_cfg.encoder:
            freeze(self.encoder)
        if self.freeze_cfg.decoder:
            freeze(self.decoder)
        if self.freeze_cfg.discriminator:            
            freeze(self.discriminator)

        self.benchmarker = Benchmarker()

        #  缓存学习率，供 configure_optimizers 使用
        self.generator_lr: float = 0.0
        self.autoencoder_lr: float = 0.0
        self.discriminator_lr: float = 0.0
        self._scale_factor = 1.0
    
    def load_state_dict(self, state_dict, strict=True):
        # 强制 non-strict，容忍旧 checkpoint 缺少任何键
        super().load_state_dict(state_dict, strict=False)

    def rescale(self, image: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Simple rescale by factor."""
        if scale_factor == 1.0:
            return image

        *leading_dims, c, h, w = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # 压成 4D
        image_4d = image.view(-1, c, h, w)

        # 插值
        image_4d = torch.nn.functional.interpolate(
            image_4d, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        # 还原成原维度
        return image_4d.view(*leading_dims, c, new_h, new_w)


    # ---------------- 属性 ---------------- #
    @property
    def scale_factor(self) -> Fraction:
        return Fraction(self.supersampling_factor, self.autoencoder.downscale_factor)

    @property
    def last_layer_weight(self) -> Tensor:
        res = self.autoencoder.last_layer_weights
        if res is None:
            res = self.decoder.last_layer_weights
        if res is None:
            res = self.encoder.last_layer_weights
        if res is None:
            raise RuntimeError("No last_layer_weights found in autoencoder/decoder/encoder")
        return res
    
    @staticmethod
    def get_scaled_size(scale: Fraction, size: Iterable[int]) -> Tuple[int, ...]:
        return tuple(round(scale * s) for s in size)

    def get_active_loss_groups(self) -> Dict[str, bool]:
        step = self.step_tracker.get_step() if self.step_tracker else 0
        return {
            "gaussian": self.gaussian_losses.is_active(step) if self.gaussian_losses else False,
            "context": self.context_losses.is_active(step),
            "target_autoencoder": self.target_autoencoder_losses.is_active(step),
            "target_render_latent": self.target_render_latent_losses.is_active(step),
            "target_render_image": self.target_render_image_losses.is_active(step),
            "target_combined": self.target_combined_losses.is_active(step),
        }
    
    # ---------------- 优化器 --------------- #
    @staticmethod
    def get_optimizer(
        optimizer_cfg: GeneratorOptimizerCfg | DiscriminatorOptimizerCfg,
        params: Iterator[Parameter] | list[Dict[str, Any]],
        lr: float,
    ) -> optim.Optimizer:
        return getattr(optim, optimizer_cfg.name)(
            params,
            lr=lr,
            **(optimizer_cfg.kwargs or {}),
        )

    @staticmethod
    def get_lr_scheduler(
        opt: optim.Optimizer,
        lr_scheduler_cfg: LRSchedulerCfg,
    ) -> optim.lr_scheduler.LRScheduler:
        return getattr(optim.lr_scheduler, lr_scheduler_cfg.name)(
            opt,
            **(lr_scheduler_cfg.kwargs or {}),
        )

    def configure_optimizers(self):
    # 现场计算 effective batch size
        eff_bs = (
            getattr(self.trainer, "num_devices", 1)
            * getattr(self.trainer, "num_nodes", 1)
            * self.trainer.datamodule.data_loader_cfg.train.batch_size
        )

        gen_lr = (
            eff_bs * self.optimizer_cfg.generator.lr
            if self.optimizer_cfg.generator.scale_lr
            else self.optimizer_cfg.generator.lr
        )
        auto_lr = (
            eff_bs * self.optimizer_cfg.generator.autoencoder_lr
            if self.optimizer_cfg.generator.scale_autoencoder_lr
            else self.optimizer_cfg.generator.autoencoder_lr
        )

        # 把计算值写回实例属性
        self.generator_lr = gen_lr
        self.autoencoder_lr = auto_lr

        optimizers = []
        schedulers = []

        # Generator optimizer
        g_opt = self.get_optimizer(
            self.optimizer_cfg.generator,
            [
                {"params": chain(self.encoder.parameters(), self.decoder.parameters())},
                {"params": self.autoencoder.parameters(), "lr": auto_lr}
                | (self.optimizer_cfg.generator.autoencoder_kwargs or {}),
            ],
            gen_lr,
        )
        optimizers.append(g_opt)
        if self.optimizer_cfg.generator.scheduler:
            schedulers.append(self.get_lr_scheduler(g_opt, self.optimizer_cfg.generator.scheduler))

        # Discriminator optimizer
        if self.discriminator:
            disc_lr = (
                eff_bs * self.optimizer_cfg.discriminator.lr
                if self.optimizer_cfg.discriminator.scale_lr
                else self.optimizer_cfg.discriminator.lr
            )
            self.discriminator_lr = disc_lr
            d_opt = self.get_optimizer(
                self.optimizer_cfg.discriminator,
                self.discriminator.parameters(),
                disc_lr,
            )
            optimizers.append(d_opt)
            if self.optimizer_cfg.discriminator.scheduler:
                schedulers.append(self.get_lr_scheduler(d_opt, self.optimizer_cfg.discriminator.scheduler))

        return optimizers, schedulers

    def forward(self, batch):
        context = batch["context"]
        features = None
        if self.encode_latents:
            posterior = self.autoencoder.encode(context["image"])
            features = posterior.sample()
        gaussians = self.encoder(
            context,
            self.step_tracker.get_step() if self.step_tracker else 0,
            features=features,
            deterministic=True
        )
        return gaussians, {}

    # ---------------- 训练步 --------------- #
    def training_step(self, batch, batch_idx: int):
        if self.step_tracker:
            self.step_tracker.set_step(self.global_step)
            self.log("step_tracker/step", self.step_tracker.get_step())

        opt = self.optimizers()
        if isinstance(opt, list):
            g_opt, d_opt = opt
        else:
            g_opt = opt

        batch = self.data_shim(batch)
        v_c = batch["context"]["image"].shape[1]
        b, v_t = batch["target"]["image"].shape[:2]
        size = self.get_scaled_size(self.scale_factor, batch["target"]["image"].shape[-2:])
        is_active_loss = self.get_active_loss_groups()

        # ---------- 初始化预测 & GT ----------
        gaussian_pred = Prediction()
        context_pred   = Prediction()
        context_gt     = GroundTruth(batch["context"]["image"])
        target_autoencoder_pred  = Prediction()
        target_autoencoder_gt    = GroundTruth(batch["target"]["image"])
        target_render_latent_pred = Prediction()
        target_render_latent_gt   = GroundTruth(near=batch["target"]["near"], far=batch["target"]["far"])
        target_render_image_pred  = Prediction()
        target_render_image_gt    = GroundTruth(
            image=self.rescale(batch["target"]["image"], self.scale_factor) if is_active_loss["target_render_image"] else None,
            near=batch["target"]["near"],
            far =batch["target"]["far"],
        )
        target_combined_pred = Prediction()
        target_combined_gt   = GroundTruth(batch["target"]["image"],
                                        near=batch["target"]["near"],
                                        far =batch["target"]["far"])

        # ---------- 生成器前向 ----------
        self.toggle_optimizer(g_opt)
        
        latents_to_decode = {}
        context_latents   = None

        # 1) context 编码
        if is_active_loss["context"] or (self.encode_latents and
                (is_active_loss["target_render_latent"] or is_active_loss["target_render_image"] or is_active_loss["target_combined"])):
            context_pred.posterior = self.autoencoder.encode(batch["context"]["image"])
            context_latents = context_pred.posterior.sample()
            if is_active_loss["context"]:
                latents_to_decode["context"] = context_latents

        # 2) target 编码
        if is_active_loss["target_autoencoder"] or is_active_loss["target_render_latent"]:
            target_autoencoder_pred.posterior = self.autoencoder.encode(batch["target"]["image"])
            target_latents = target_autoencoder_pred.posterior.sample()
            if is_active_loss["target_autoencoder"]:
                latents_to_decode["target"] = target_latents
            if is_active_loss["target_render_latent"]:
                target_render_latent_gt.image = target_latents

        # 3) 高斯编码 + 渲染
        if any(is_active_loss[k] for k in ("gaussian", "target_render_latent",
                                        "target_render_image", "target_combined")):
            gaussians: VariationalGaussians = self.encoder(
                batch["context"],
                self.step_tracker.get_step(),
                features=context_latents if self.encode_latents else None,
                deterministic=False
            )
            if is_active_loss["gaussian"]:
                gaussian_pred.posterior = gaussians.feature_harmonics

            output = self.decoder.forward(
                gaussians.sample() if self.variational in ("gaussians", "none") else gaussians.flatten(),
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                size,
                depth_mode=self.train_cfg.depth_mode,
                return_colors=is_active_loss["target_render_image"],
                return_features=is_active_loss["target_render_latent"] or is_active_loss["target_combined"],
            )

            # ✅ 只 detach 你需要的字段（假设是 image 和 color）
            if hasattr(output, 'image') and output.image is not None:
                output.image = output.image.detach().requires_grad_()
            if hasattr(output, 'color') and output.color is not None:
                output.color = output.color.detach().requires_grad_()
            if hasattr(output, 'features') and output.features is not None:
                output.features = output.features.detach().requires_grad_()

            target_render_image_pred.image = output.color
            target_render_latent_pred.posterior = output.feature_posterior
            latent_sample = output.feature_posterior.sample()
            z = self.rescale(latent_sample, Fraction(1, self.supersampling_factor))
            target_render_latent_pred.image = z

            if is_active_loss["target_combined"]:
                skip_z = None
                if self.autoencoder.expects_skip:
                    skip_z = torch.cat((output.color.detach(), latent_sample), dim=-3) \
                        if self.autoencoder.expects_skip_extra else latent_sample
                target_combined_pred.image = self.autoencoder.decode(z, skip_z)

        # 4) 批量解码 latents
        if latents_to_decode:
            split_sizes = [prod(l.shape[:-3]) for l in latents_to_decode.values()]
            latents = torch.cat([l.flatten(0, -4) for l in latents_to_decode.values()])
            images = self.autoencoder.decode(latents)
            pred_images = dict(zip(latents_to_decode.keys(), images.split(split_sizes)))
            if is_active_loss["context"]:
                context_pred.image = rearrange(pred_images["context"],
                                            "(b v) c h w -> b v c h w", b=b, v=v_c)
            if is_active_loss["target_autoencoder"]:
                target_autoencoder_pred.image = rearrange(pred_images["target"],
                                                        "(b v) c h w -> b v c h w", b=b, v=v_t)

        # ---------- 指标日志 ----------
        for view, pred, gt in zip(
                ("context", "target_autoencoder", "target_render", "target_combined"),
                (context_pred, target_autoencoder_pred, target_render_image_pred, target_combined_pred),
                (context_gt, target_autoencoder_gt, target_render_image_gt, target_combined_gt)):
            if gt.image is not None and pred.image is not None:
                psnr = compute_psnr(
                    rearrange(gt.image, "b v c h w -> (b v) c h w"),
                    rearrange(pred.image, "b v c h w -> (b v) c h w"))
                self.log(f"train/{view}/psnr", psnr.mean())

        # ---------- 判别器 logits ----------
        for loss_group, pred in zip(
                (self.context_losses, self.target_autoencoder_losses, self.target_combined_losses),
                (context_pred, target_autoencoder_pred, target_combined_pred)):
            if loss_group.is_generator_loss_active(self.step_tracker.get_step()):
                b, v = pred.image.shape[:2]
                logits_fake = self.discriminator(
                    rearrange(pred.image, "b v c h w -> (b v) c h w"))
                pred.logits_fake = rearrange(logits_fake, "(b v) c h w -> b v c h w", b=b, v=v)

        # ---------- 生成器损失 ----------
        generator_loss = 0.
        for loss_group, pred, gt in zip(
                (self.gaussian_losses, self.context_losses, self.target_autoencoder_losses,
                self.target_render_image_losses, self.target_render_latent_losses, self.target_combined_losses),
                (gaussian_pred, context_pred, target_autoencoder_pred,
                target_render_image_pred, target_render_latent_pred, target_combined_pred),
                (None, context_gt, target_autoencoder_gt,
                target_render_image_gt, target_render_latent_gt, target_combined_gt)):
            if loss_group is None:
                continue
            group_loss, loss_dict = loss_group.forward_generator(
                pred, gt, self.step_tracker.get_step(), self.last_layer_weight)
            for loss_name, loss in loss_dict.items():
                self.log(f"loss/generator/{loss_name}", loss.unweighted)
            self.log(f"loss/generator/{loss_group.name}/total", group_loss)
            generator_loss = generator_loss + group_loss

        if self.gaussian_losses is not None:
            _, gaussian_loss_dict = self.gaussian_losses.forward_generator(
                gaussian_pred, None, self.step_tracker.get_step(), self.last_layer_weight)
        else:
            log("gaussian_losses is None, skipping forward_generator")

        # ---------- 生成器反向 ----------
        if isinstance(generator_loss, Tensor) and not generator_loss.isnan().any():
            g_opt.zero_grad()
            self.manual_backward(generator_loss)
            self.clip_gradients(
                g_opt,
                gradient_clip_val=self.optimizer_cfg.generator.gradient_clip_val,
                gradient_clip_algorithm=self.optimizer_cfg.generator.gradient_clip_algorithm)
            g_opt.step()
        else:
            warn(f"NaN generator_loss at step {self.step_tracker.get_step()}")

        self.untoggle_optimizer(g_opt)

        # ---------- 判别器损失 & 反向 ----------
        if self.discriminator is not None:
            self.toggle_optimizer(d_opt)
            discriminator_loss = 0.
            for loss_group, pred, gt in zip(
                    (self.context_losses, self.target_autoencoder_losses, self.target_combined_losses),
                    (context_pred, target_autoencoder_pred, target_combined_pred),
                    (context_gt, target_autoencoder_gt, target_combined_gt)):
                if loss_group.is_discriminator_loss_active(self.step_tracker.get_step()):
                    b, v = pred.image.shape[:2]
                    logits_fake = self.discriminator(
                        rearrange(pred.image.detach(), "b v c h w -> (b v) c h w"))
                    logits_real = self.discriminator(
                        rearrange(gt.image, "b v c h w -> (b v) c h w"))
                    pred.logits_fake = rearrange(logits_fake, "(b v) c h w -> b v c h w", b=b, v=v)
                    pred.logits_real = rearrange(logits_real, "(b v) c h w -> b v c h w", b=b, v=v)
                    group_loss, loss_dict = loss_group.forward_discriminator(
                        pred, gt, self.step_tracker.get_step())
                    for loss_name, loss in loss_dict.items():
                        self.log(f"loss/discriminator/{loss_name}", loss.unweighted)
                    self.log(f"loss/discriminator/{loss_group.name}/total", group_loss)
                    discriminator_loss = discriminator_loss + group_loss

            self.log("loss/discriminator/total", discriminator_loss)

            if isinstance(discriminator_loss, Tensor) and not discriminator_loss.isnan().any():
                d_opt.zero_grad()
                self.manual_backward(discriminator_loss)
                self.clip_gradients(
                    d_opt,
                    gradient_clip_val=self.optimizer_cfg.discriminator.gradient_clip_val,
                    gradient_clip_algorithm=self.optimizer_cfg.discriminator.gradient_clip_algorithm)
                d_opt.step()
            else:
                warn(f"NaN discriminator_loss at step {self.step_tracker.get_step()}")

            self.untoggle_optimizer(d_opt)
        else:
            discriminator_loss = None

        # ---------- 进度 & scheduler ----------
        if self.global_rank == 0:
            progress = (f"train step {self.step_tracker.get_step()}; "
                        f"scene = {batch['scene']}; "
                        f"context = {batch['context']['index'].tolist()}; "
                        f"generator loss = {generator_loss:.6f}")
            if discriminator_loss is not None:
                progress += f"; discriminator loss = {discriminator_loss:.6f}"

        self.log("loss/generator/total", generator_loss, prog_bar=True)

        schedulers = self.lr_schedulers()
        if schedulers:
            if isinstance(schedulers, list):
                for sch in schedulers:
                    sch.step()
            else:
                schedulers.step()

    # 粘到 ModelWrapper 类里
    def build_spiral_camera(self, b, azim_deg, elev_deg=20., dist=2.):
        """返回虚拟相机外参 tensor [b, 4, 4]"""
        import math
        azim = math.radians(azim_deg)
        elev = math.radians(elev_deg)
        cam_pos = torch.tensor([dist * math.cos(elev) * math.sin(azim),
                                dist * math.sin(elev),
                                dist * math.cos(elev) * math.cos(azim)],
                              device=self.device)
        target = torch.zeros(3, device=self.device)
        up = torch.tensor([0., 1., 0.], device=self.device)

        # look-at
        z = (cam_pos - target) / torch.norm(cam_pos - target)
        x = torch.cross(up, z)
        x /= x.norm()
        y = torch.cross(z, x)
        pose = torch.eye(4, device=self.device)
        pose[:3, 0] = x
        pose[:3, 1] = y
        pose[:3, 2] = z
        pose[:3, 3] = cam_pos
        return pose.unsqueeze(0).repeat(b, 1, 1)   # [b,4,4]

    # ---------------- 验证 / 测试 / 视频 --------------- #
    @rank_zero_only
    def validation_step(self, batch: BatchedExample, batch_idx: int):
        if self.global_rank == 0:
            print(
                f"validation step {self.step_tracker.get_step()}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        b, v = batch["target"]["image"].shape[:2]
        assert b == 1
        size = self.get_scaled_size(self.scale_factor, batch["target"]["image"].shape[-2:])

        pred = {"low": {}, "high": {}}

        # ---------- 1. 概率前向 ----------
        if self.encode_latents:
            posterior = self.autoencoder.encode(batch["context"]["image"])
            context_latents = posterior.sample()
        else:
            context_latents = None

        gaussians_prob: VariationalGaussians = self.encoder(
            batch["context"],
            self.step_tracker.get_step(),
            features=context_latents,
            deterministic=False,
        )
        output_prob = self.decoder.forward(
            gaussians_prob.sample() if self.variational in ("gaussians", "none") else gaussians_prob.flatten(),
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            size,
        )
        pred["low"]["probabilistic"] = output_prob.color[0]
        latent_prob = output_prob.feature_posterior.sample()[0]

        # ---------- 2. 确定性前向 ----------
        gaussians_det: VariationalGaussians = self.encoder(
            batch["context"],
            self.step_tracker.get_step(),
            features=posterior.mode() if self.encode_latents else None,
            deterministic=True,
        )
        output_det = self.decoder.forward(
            gaussians_det.mode() if self.variational in ("gaussians", "none") else gaussians_prob.flatten(),
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            size,
        )
        pred["low"]["deterministic"] = output_det.color[0]
        latent_det = output_det.feature_posterior.mode()[0]

        # ---------- 3. 高分辨率解码 ----------
        latents = torch.cat([latent_prob, latent_det])
        z = self.rescale(latents, Fraction(1, self.supersampling_factor))

        if self.autoencoder.expects_skip:
            if self.autoencoder.expects_skip_extra:
                colors = torch.cat([pred["low"]["probabilistic"], pred["low"]["deterministic"]])
                skip_z = torch.cat([colors, latents], dim=-3)
            else:
                skip_z = latents
        else:
            skip_z = None

        dec = self.autoencoder.decode(z, skip_z)
        pred["high"]["probabilistic"], pred["high"]["deterministic"] = dec.tensor_split(2)

        # ---------- 4. 指标计算 ----------
        rgb_high_res_gt = batch["target"]["image"][0]
        rgb_low_res_gt  = self.rescale(rgb_high_res_gt, self.scale_factor)

        for mode in ("deterministic", "probabilistic"):
            score = compute_psnr(rgb_low_res_gt, pred["low"][mode]).mean()
            self.log(f"val/{mode}/low/psnr", score, rank_zero_only=True)

            for metric_name, metric in zip(
                ("psnr", "lpips", "ssim"),
                (compute_psnr, compute_lpips, compute_ssim),
            ):
                score = metric(rgb_high_res_gt, pred["high"][mode]).mean()
                self.log(f"val/{mode}/high/{metric_name}", score, rank_zero_only=True)

        # ---------- 5. 可视化 ----------
        comparison_low = add_border(
            hcat(
                vcat(*rgb_low_res_gt, gap=1),
                vcat(*pred["low"]["probabilistic"], gap=1),
                vcat(*pred["low"]["deterministic"], gap=1),
                gap=1,
            ),
            border=1,
        )
        comparison_high = add_border(
            hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_high_res_gt), "Target (Ground Truth)"),
                add_label(vcat(*pred["high"]["probabilistic"]), "Target (Probabilistic)"),
                add_label(vcat(*pred["high"]["deterministic"]), "Target (Deterministic)"),
            )
        )

        self.logger.log_image(
            "comparison_low", [prep_image(comparison_low)], step=self.step_tracker.get_step(), caption=batch["scene"]
        )
        self.logger.log_image(
            "comparison_high", [prep_image(comparison_high)], step=self.step_tracker.get_step(), caption=batch["scene"]
        )

        # ---------- 6. 相机轨迹视频 ----------
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image("cameras", [prep_image(add_border(cameras))], step=self.step_tracker.get_step())

        # ---------- 7. 视频生成 ----------
        if self.train_cfg.video_interpolation:
            self.render_video_interpolation(batch)
        if self.train_cfg.video_wobble:
            self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    def test_step(self, batch: BatchedExample, batch_idx: int) -> None:
        batch = self.data_shim(batch)
        b, v = batch["target"]["image"].shape[:2]
        size = self.get_scaled_size(1.0, batch["target"]["image"].shape[-2:])
        assert b == 1

        # ---------- 1. 编码 ----------
        if self.encode_latents:
            posterior = self.autoencoder.encode(batch["context"]["image"])
            context_latents = posterior.sample()
        else:
            context_latents = None

        # ---------- 2. 渲染 ----------
        gaussians: VariationalGaussians = self.encoder(
            batch["context"],
            self.step_tracker.get_step(),
            features=context_latents,
            deterministic=False,
        )
        output = self.decoder.forward(
            gaussians.sample() if self.variational in ("gaussians", "none") else gaussians.flatten(),
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            size,
        )

        # ---------- 3. 解码 ----------
        latent_sample = output.feature_posterior.sample()
        z = self.rescale(latent_sample, Fraction(1, self.supersampling_factor))

        # 1. 投影到 VAE 期望的通道数
        if z.shape[-3] != self.autoencoder.latent_channels:
            # 临时 1×1 卷积，权重随推理即可（也可换成 nn.Conv2d 注册）
            z = torch.nn.functional.conv2d(
                z,
                weight=torch.randn(
                    self.autoencoder.latent_channels,
                    z.shape[-3], 1, 1,
                    device=z.device,
                    dtype=z.dtype,
                ),
                bias=None,
                stride=1,
                padding=0,
            )

        skip_z = (
            torch.cat((output.color.detach(), latent_sample), dim=-3)
            if self.autoencoder.expects_skip_extra
            else latent_sample
        ) if self.autoencoder.expects_skip else None

        # 2. 后续插值、解码保持原逻辑
        spatial_ndim = z.ndim - 2
        spatial = z.shape[-spatial_ndim:]
        target_spatial = tuple(round(s * 4) for s in spatial)
        mode = "trilinear" if spatial_ndim == 3 else "bilinear"

        z_big = torch.nn.functional.interpolate(
            z, size=target_spatial, mode=mode, align_corners=False
        )
        print("z_big.shape:", z_big.shape)
        if skip_z is None:
            torch.cuda.empty_cache()
            # 5 维已压成 4 维 (B,V,C,H,W)
            b, v, c, h, w = z_big.shape
            tile = self.cfg.test.decode_tile          # 读配置
            if tile <= 0 or h <= tile and w <= tile:  # 整图模式（A100）
                with torch.no_grad():
                    target_pred_big = self.autoencoder.decode(z_big, None).sample
            else:                                       # 分块模式（小卡）
                target_pred_big = torch.zeros_like(z_big[:, :, :3])
                for i in range(0, h, tile):
                    for j in range(0, w, tile):
                        z_tile = z_big[:, :, :, i:i+tile, j:j+tile]
                        with torch.no_grad():
                            pred_tile = self.autoencoder.decode(z_tile, None).sample
                        target_pred_big[:, :, :, i:i+tile, j:j+tile] = pred_tile
                        del pred_tile
                torch.cuda.empty_cache()

        # 3. 提亮
        target_pred_image = (target_pred_image - target_pred_image.min()).clamp_min(0) / \
                            (target_pred_image.max() - target_pred_image.min()).clamp_min(1e-5)

        # ---------- 4. 保存 ----------
        (scene,) = batch["scene"]
        context_index_str = "_".join(map(str, sorted(batch["context"]["index"][0].tolist())))
        path = Path(self.test_cfg.output_path) / "debug" / scene / context_index_str / "color"
        path.mkdir(parents=True, exist_ok=True)
        for index, color in zip(batch["target"]["index"][0], target_pred_image[0]):
            save_image(color, path / f"{index:0>6}.png")

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(self.test_cfg.output_path / name / "peak_memory.json")