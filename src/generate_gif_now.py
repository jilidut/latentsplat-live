#generate_gif_now.py
"""
generate_gif_now.py  （仅保留关键改动，其余逻辑与你原来保持一致）
"""
import sys
from pathlib import Path
from fractions import Fraction
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl

# ---------- 原 import 列表 ----------
from src.model.autoencoder import get_autoencoder
from src.model.encoder     import get_encoder
from src.model.decoder     import get_decoder
from src.dataset.data_module import DataModule
from src.model.model_wrapper import ModelWrapper
from src.callbacks.generate_gif import GenerateGif

# -------------------------------------------------
# 1. 加载 checkpoint 与对应 cfg
# -------------------------------------------------
# ckpt_path = Path(sys.argv[1]).expanduser().resolve()
ckpt_path = Path(sys.argv[1]).expanduser().resolve()
ckpt      = torch.load(ckpt_path, map_location="cpu")
# cfg_path  = ckpt_path.parent / "config.yaml"
cfg_path = Path("/home/jilidut/download/file/latentsplat/latentsplat-main/outputs/2025-09-07/17-28-53.485792/.hydra/config.yaml")
cfg       = OmegaConf.load(cfg_path)

# 允许动态写字段
OmegaConf.set_struct(cfg, False)

# -------------------------------------------------
# 2. 一次性补全所有缺失字段（dataset / encoder / decoder / autoencoder）
# -------------------------------------------------
# 2.1 dataset
dset_cfg = cfg.dataset
dset_cfg.setdefault("patch_size", 0)          # 关 patch

# 2.2 autoencoder
auto_cfg = cfg.model.autoencoder
auto_cfg.setdefault("pretrained", False)
auto_cfg.setdefault("skip_connections", False)
auto_cfg.setdefault("skip_extra", False)
auto_cfg.setdefault("skip_zero", False)

# 2.3 encoder
enc_cfg = cfg.model.encoder
enc_cfg.gaussians_per_pixel = 1   # 降压：每像素只出 1 个高斯
enc_cfg.setdefault("patch_size", 0)           # 关 patch
enc_cfg.setdefault("pretrained", False)
enc_cfg.setdefault("n_feature_channels", 64)
enc_cfg.backbone.upscale_mode = "repeat"
enc_cfg.backbone.setdefault("upscale_mode", "bilinear")
enc_cfg.epipolar_transformer.setdefault("num_context_views", 2)
enc_cfg.epipolar_transformer.setdefault("num_samples", 32)

# 2.4 decoder
dec_cfg = cfg.model.decoder
dec_cfg.setdefault("pretrained", False)
# dec_cfg.name = "splatting_soft"

# -------------------------------------------------
# 2.5 统一给所有 loss 子项补缺失的 generator / discriminator
# -------------------------------------------------
# 先拿到 loss 配置
loss_cfg = cfg.loss

# 再定义函数
def _safe_add_gd(cfg_node):
    cfg_node.setdefault("generator",      None)
    cfg_node.setdefault("discriminator",  None)

# 只给确实存在的节点补 generator / discriminator
def safe_add_gd(path: str):
    """path 用点号分隔，例如 'target.render.image'"""
    node = loss_cfg
    for key in path.split('.'):
        if key not in node:
            return          # 路径不存在，直接跳过
        node = node[key]
    _safe_add_gd(node)

# 逐个路径补
safe_add_gd('gaussian')
safe_add_gd('target.render.image')
safe_add_gd('target.render.latent')   # 如果不存在就自动跳过
safe_add_gd('target.combined')
safe_add_gd('target.autoencoder')

# -------------------------------------------------
# 3. 手动 build 子模块（只用这份 cfg，不再重新 instantiate）
# -------------------------------------------------
autoencoder = get_autoencoder(auto_cfg)

encoder, encoder_visualizer = get_encoder(
    enc_cfg,
    d_in=autoencoder.d_latent if cfg.model.encode_latents else 3,
    n_feature_channels=autoencoder.d_latent,
    scale_factor=Fraction(
        cfg.model.supersampling_factor,
        1 if cfg.model.encode_latents else autoencoder.downscale_factor,
    ),
    variational=cfg.model.variational != "none",
)

decoder = get_decoder(
    dec_cfg,
    cfg.dataset.background_color,
    cfg.model.variational == "latents",
)

# -------------------------------------------------
# 4. 实例化 DataModule 与 ModelWrapper
# -------------------------------------------------
dm = DataModule(cfg, cfg.data_loader) # 把已有 autoencoder 传进去，避免内部再 build

model = ModelWrapper(
    cfg,
    cfg.optimizer,
    cfg.test,
    cfg.train,
    cfg.freeze,
    autoencoder,                      # 第 6
    encoder,                          # 第 7
    cfg.model.encode_latents,         # 第 8
    encoder_visualizer,               # 第 9
    decoder,                          # 第 10
    supersampling_factor=cfg.model.get("supersampling_factor", 1),
    variational=cfg.model.variational,
    discriminator=None,               # 没有就 None
    context_loss_cfg=None,
    target_autoencoder_loss_cfg=None,
    target_render_latent_loss_cfg=None,
    target_combined_loss_cfg=None,
    target_render_image_loss_cfg=None,
    step_tracker=None,
)

# 加载权重
model.load_state_dict(ckpt["state_dict"])
model = model.cuda()

# -------------------------------------------------

# ---------- 绕过 patch_shim ----------
from src.dataset.shims import patch_shim
patch_shim.apply_patch_shim = lambda batch, patch_size: batch   # 原样返回
patch_shim.apply_patch_shim_to_views = lambda views, patch_size: views
from src.dataset.shims import bounds_shim
bounds_shim.apply_bounds_shim = lambda batch, nd, fd: batch   # 原样返回
# ---------- 绕过两个 shim ----------
from src.dataset.shims import patch_shim, bounds_shim
patch_shim.apply_patch_shim = lambda batch, patch_size: batch
patch_shim.apply_patch_shim_to_views = lambda views, patch_size: views
bounds_shim.apply_bounds_shim = lambda batch, near_disp, far_disp: batch
# ---------- 强制替换已导入的函数 ----------
import src.dataset.shims.bounds_shim as bs
bs.apply_bounds_shim = lambda batch, near_disp, far_disp: batch

import src.dataset.shims.patch_shim as ps
ps.apply_patch_shim = lambda batch, patch_size: batch
ps.apply_patch_shim_to_views = lambda views, patch_size: views

# 5. Trainer + 生成 GIF
# -------------------------------------------------
trainer = pl.Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger=False,
)
gif_cb = GenerateGif(
    ckpt_path="last",
    output_dir=ckpt_path.parent / "gifs",
    fps=12,
    n_frames=60
)
# -------------------------------------------------
# 5. 手动构造 batch 并生成 GIF
# -------------------------------------------------
import torch
from src.callbacks.generate_gif import GenerateGif

# 1. 构造假相机参数（与你训练时同尺寸即可）
# 1. 基础配置
B, N = 1, 2
H, W = 128, 128
fx = fy = 75.6 * (128 / 180)
cx, cy = 64.0, 64.0

K = torch.tensor([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
extr = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
near = torch.ones(N) * 0.1
far  = torch.ones(N) * 10.0
rgb  = torch.rand(N, 3, H, W)

# 1. 先拼 context
batch = {
    "context": {
        "image": rgb.unsqueeze(0),
        "extrinsics": extr.unsqueeze(0),
        "intrinsics": K.unsqueeze(0),
        "near": near.unsqueeze(0),
        "far":  far.unsqueeze(0),
        "index": torch.arange(N).unsqueeze(0),
    },
    "target": {},  # 占位，后面再补
    "scene": [],   # 占位，后面再补
}

# 2. 再拼 target（此时 batch 已存在）
batch["target"] = {
    "image": batch["context"]["image"].clone(),
    "extrinsics": batch["context"]["extrinsics"].clone(),
    "intrinsics": batch["context"]["intrinsics"].clone(),
    "near": batch["context"]["near"].clone(),
    "far": batch["context"]["far"].clone(),
    "index": batch["context"]["index"].clone(),
}

# 3. 最后把 scene 指向自己
batch["scene"] = [batch]
batch["target"] = {
    "image": batch["context"]["image"].clone(),
    "extrinsics": batch["context"]["extrinsics"].clone(),
    "intrinsics": batch["context"]["intrinsics"].clone(),
    "near": batch["context"]["near"].clone(),
    "far": batch["context"]["far"].clone(),
    "index": batch["context"]["index"].clone(), 
}
class MockStepTracker:
    def get_step(self):
        return 0
    
model = model.cuda()          # 整个网络搬 CUDA
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
         for k, v in batch.items()}  # 输入也搬 CUDA

model.step_tracker = MockStepTracker()

# 把模型和整个 batch 都搬 CUDA
model = model.cuda()

def to_cuda(obj, _depth=0, max_depth=20):
    if _depth > max_depth:
        return obj  # 防止无限深递归
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cuda(v, _depth + 1) for v in obj)
    if isinstance(obj, dict):
        return {k: to_cuda(v, _depth + 1) for k, v in obj.items()}
    # 忽略 OmegaConf 结构、自定义类、None 等
    return obj

batch = to_cuda(batch)

with torch.no_grad():
    out = model.test_step(batch, 0)

# 从输出里取出高斯点
gaussians = out['gaussians']  # 或 out.gaussians，看你模型返回结构

# 把裁剪后的高斯点重新塞回去（可选，看后续是否需要）
out['gaussians'] = gaussians

gif_cb = GenerateGif(
    ckpt_path="last",
    output_dir=ckpt_path.parent / "gifs",
    fps=12,
    n_frames=60
)
gif_cb.visualize_batch(out, batch, model, ckpt_path.parent / "gifs")
print("GIF 已生成到：", ckpt_path.parent / "gifs")