# main.py
from fractions import Fraction
import os
import time
from pathlib import Path
import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from src.callbacks.gif_every_n_step import GifEveryNStep
from src.callbacks.generate_gif import GenerateGif
# Configure beartype and jaxtyping.
# with install_import_hook(
#     ("src",),
#     ("beartype", "beartype"),
# ):
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

torch.set_float32_matmul_precision('medium')

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)



# @hydra.main(version_base=None, config_path="../config", config_name="main")
@hydra.main(config_path="../config", config_name="main", version_base=None)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    from omegaconf import OmegaConf
    print("========== cfg_dict.loss ==========")
    print(OmegaConf.to_yaml(cfg_dict.loss)) 

    print(f"[DEBUG] Hydra config loaded from: {hydra.core.hydra_config.HydraConfig.get().job.config_name}")
    print(f"[DEBUG] Full config:\n{OmegaConf.to_yaml(cfg_dict)}")

    # Set up the output directory.
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.activated:
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode="offline",
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            entity=cfg_dict.wandb.entity,
        )
        callbacks.append(LearningRateMonitor("step", True))
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    # 1. 每 2500 步留一个快照，最多 10 个（防止炸盘）
    callbacks = []

    # 1. 定时快照 + last.ckpt
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints" / "snapshots",
            filename="step{step:06d}",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=-1,
            save_last=True,
            auto_insert_metric_name=False,
        )
    )

    # 2. 验证最优
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints" / "best",
            filename="best-step{step:06d}-val_loss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        )
    )


    callbacks[-2].save_last = True 

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    step_tracker = StepTracker(cfg.train.step_offset)


    # 新代码（直接粘贴上去）
    callbacks.append(
        GifEveryNStep(every_n_steps=50,        # 测试时设 50，正式训练改 2500
                    output_dir=output_dir / "progress_gif",
                    fps=12, n_frames=20, h=176, w=176)
    )

    log(">>> 即将构建 Trainer")
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=[
            TensorBoardLogger("outputs", name="lightning_logs"),
            logger,  # 这是你原来的 WandbLogger
        ],
        devices="auto",
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
    )

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

    print(f"[DEBUG] cfg.loss = {cfg.loss}")
    print(f"[DEBUG] cfg.loss.gaussian = {cfg.loss.gaussian}")

    kwargs = dict(
        cfg=cfg,
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
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path if cfg.checkpointing.resume else None)
        log(">>> trainer.fit 已返回")
    elif cfg.mode == "val":
        trainer.validate(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    elif cfg.mode == "test":
        trainer.test(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    train()