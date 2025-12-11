#src/main.py
from fractions import Fraction
import os
import time
from pathlib import Path

import hydra
import torch
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.callbacks.gif_every_n_step import GifEveryNStep
import torch
# torch.cuda.set_per_process_memory_fraction(0.6)  # ⬅️ 每个进程最多用50%
torch.cuda.set_per_process_memory_fraction(0.95, 0)  # 留 5 % 余量

# 启用 beartype & jaxtyping
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.autoencoder import get_autoencoder
    from src.model.decoder import get_decoder
    from src.model.discriminator import get_discriminator
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

torch.cuda.set_per_process_memory_fraction(0.95, 0)
torch.set_float32_matmul_precision("medium")


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    log(cyan(f"Saving outputs to {output_dir}"))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm -f {latest_run} && ln -s {output_dir} {latest_run}")

    # -------------- logger & callbacks（仅保留 TB + GIF） --------------
    callbacks = []

    # 1. checkpoint
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="step{step:06d}",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=-1,
            monitor=None,
            save_last=True,
            auto_insert_metric_name=False,
        )
    )

    # 2. GIF 进度
    callbacks.append(
        GifEveryNStep(
            every_n_steps=25_000,
            output_dir=output_dir / "progress_gif",
            fps=12,
            n_frames=20,
            h=176,
            w=176,
        )
    )

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    step_tracker = StepTracker(cfg.train.step_offset)

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=TensorBoardLogger(save_dir=output_dir, name="lightning_logs"),
        devices="auto",
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
    )

    # ---------------- 模型构建 ----------------
    autoencoder = get_autoencoder(cfg.model.autoencoder)
    encoder, encoder_visualizer = get_encoder(
        cfg.model.encoder,
        d_in=autoencoder.d_latent if cfg.model.encode_latents else 3,
        n_feature_channels=autoencoder.d_latent,
        scale_factor=Fraction(
            cfg.model.supersampling_factor,
            1 if cfg.model.encode_latents else autoencoder.downscale_factor,
        ),
        variational=cfg.model.variational != "none",
    )
    decoder = get_decoder(cfg.model.decoder, cfg.dataset.background_color, cfg.model.variational == "latents")

    kwargs = dict(
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        freeze_cfg=cfg.freeze,
        autoencoder=autoencoder,
        encoder=encoder,
        encode_latents=cfg.model.encode_latents,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        supersampling_factor=cfg.model.supersampling_factor,
        variational=cfg.model.variational,
        discriminator=get_discriminator(cfg.model.discriminator) if cfg.model.discriminator is not None else None,
        gaussian_loss_cfg=cfg.loss.gaussian,
        context_loss_cfg=cfg.loss.context,
        target_autoencoder_loss_cfg=cfg.loss.target.autoencoder,
        target_render_latent_loss_cfg=cfg.loss.target.render.latent,
        target_render_image_loss_cfg=cfg.loss.target.render.image,
        target_combined_loss_cfg=cfg.loss.target.combined,
        step_tracker=step_tracker,
    )

    log(">>> 即将实例化 ModelWrapper")
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        model_wrapper = ModelWrapper.load_from_checkpoint(checkpoint_path, **kwargs, strict=False)
    else:
        model_wrapper = ModelWrapper(**kwargs)

    log(">>> 即将构造 DataModule")
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)

    if cfg.mode == "train":
        log(">>> 即将调用 trainer.fit(...)")
        start_time = time.time()  # 开始计时
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path if cfg.checkpointing.resume else None,
        )
        # log(">>> trainer.fit 已返回")
        end_time = time.time()  # 结束计时
        total_time = end_time - start_time

        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        log(f">>> trainer.fit 已返回，训练总用时: {hours}h {minutes}m {seconds}s")

    elif cfg.mode == "val":
        trainer.validate(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)

    # elif cfg.mode == "test":
    #     trainer.test(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)

    elif cfg.mode == "test":
        # === 启用指标计算 ===
        if hasattr(cfg, "test") and hasattr(cfg.test, "compute_metrics"):
            cfg.test.compute_metrics = True
            log(">>> 已启用测试指标计算（PSNR / SSIM / LPIPS）")

        # === 执行测试 ===
        trainer.test(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)

        # === 打印与保存指标 ===
        if hasattr(model_wrapper, "benchmarker") and hasattr(model_wrapper.benchmarker, "metrics"):
            log("\n=== Test Metrics (from ModelWrapper) ===")
            metrics = model_wrapper.benchmarker.metrics
            for key in ["PSNR", "SSIM", "LPIPS"]:
                val = metrics.get(key, "N/A")
                print(f"{key}: {val}")
            log("========================================")

            # === 自动保存到 CSV ===
            import csv
            csv_path = Path("outputs/latest-run/metrics.csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["run_name", "PSNR", "SSIM", "LPIPS"])
                run_name = getattr(cfg.wandb, "name", "unnamed_run")
                writer.writerow([
                    run_name,
                    metrics.get("PSNR", "N/A"),
                    metrics.get("SSIM", "N/A"),
                    metrics.get("LPIPS", "N/A")
                ])
            log(f"✅ 指标已保存到 {csv_path}")

    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    train()