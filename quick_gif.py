# quick_gif.py
import torch, imageio, math, os
from pathlib import Path
from omegaconf import OmegaConf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = Path("outputs/2025-09-08/13-20-52.447096/checkpoints/snapshots/step195000.ckpt")

# ---- 以下 5 行照抄你 generate_gif_now.py 里已经跑通的 ----
ckpt = torch.load(ckpt_path, map_location="cpu")
cfg_path = ckpt_path.parent.parent.parent / ".hydra" / "config.yaml"
cfg = OmegaConf.load(cfg_path)
OmegaConf.set_struct(cfg, False)

auto_cfg = cfg.model.autoencoder
auto_cfg.setdefault("pretrained", False)
# auto_cfg.setdefault("skip_connections", False)
# auto_cfg.setdefault("skip_extra", False)
# auto_cfg.setdefault("skip_zero", False)

# 2.3 encoder
enc_cfg = cfg.model.encoder
# enc_cfg.gaussians_per_pixel = 1   # 降压：每像素只出 1 个高斯
# enc_cfg.setdefault("patch_size", 0)           # 关 patch
enc_cfg.setdefault("pretrained", False)
enc_cfg.setdefault("n_feature_channels", 64)
# enc_cfg.backbone.upscale_mode = "repeat"
# enc_cfg.backbone.setdefault("upscale_mode", "bilinear")
enc_cfg.backbone.setdefault("upscale_mode", "repeat")
# enc_cfg.epipolar_transformer.setdefault("num_context_views", 2)
# enc_cfg.epipolar_transformer.setdefault("num_samples", 32)
enc_cfg = cfg.model.encoder
enc_cfg.setdefault("patch_size", 0)

# 新增：补 epipolar_transformer
et_cfg = enc_cfg.epipolar_transformer
et_cfg.setdefault("num_context_views", 2)
et_cfg.setdefault("num_samples", 32)

# 2.4 decoder
dec_cfg = cfg.model.decoder
dec_cfg.setdefault("pretrained", False)
# dec_cfg.name = "splatting_soft"


# from src.generate_gif_now import _safe_add_gd
from src.model.autoencoder import get_autoencoder
from src.model.encoder import get_encoder
from src.model.decoder import get_decoder
from src.model.model_wrapper import ModelWrapper

def _safe_add_gd(cfg_root, path: str):
    """path 用点号分隔，如 'target.render.latent'"""
    node = cfg_root
    for key in path.split('.'):
        if key not in node:
            return  # 路径不存在，直接跳过
        node = node[key]
    node.setdefault("generator", None)
    node.setdefault("discriminator", None)

# 只补确实存在的节点
_safe_add_gd(cfg.loss, "gaussian")
_safe_add_gd(cfg.loss.target.render, "image")
_safe_add_gd(cfg.loss.target.render, "latent")   # 如果不存在就自动跳过
_safe_add_gd(cfg.loss.target, "combined")
_safe_add_gd(cfg.loss.target, "autoencoder")


autoencoder = get_autoencoder(cfg.model.autoencoder)
encoder, encoder_visualizer = get_encoder(
    cfg.model.encoder,
    d_in=autoencoder.d_latent if cfg.model.encode_latents else 3,
    n_feature_channels=autoencoder.d_latent,
    scale_factor=1,
    variational=cfg.model.variational != "none",
)
decoder = get_decoder(
    cfg.model.decoder,
    cfg.dataset.background_color,
    cfg.model.variational == "latents",
)

model = ModelWrapper(
    cfg, cfg.optimizer, cfg.test, cfg.train, cfg.freeze,
    autoencoder, encoder, cfg.model.encode_latents, encoder_visualizer, decoder,
)
# model.load_state_dict(ckpt["state_dict"])
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval().to(device)
# ----------------------------------------------------------

# 下面渲染代码不变
...

# 2. 构造 360° 相机
def build_camera(azim_deg, elev=20., dist=2.0, h=176, w=176):
    azim = math.radians(azim_deg)
    elev = math.radians(elev)
    cam_pos = torch.tensor([
        dist * math.cos(elev) * math.sin(azim),
        dist * math.sin(elev),
        dist * math.cos(elev) * math.cos(azim)
    ], device=device)
    target = torch.zeros(3, device=device)
    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    z = torch.nn.functional.normalize(cam_pos - target, dim=-1)
    x = torch.nn.functional.normalize(torch.cross(up, z), dim=-1)
    y = torch.cross(z, x)
    R = torch.stack([x, y, z], dim=-1).unsqueeze(0)  # [1,3,3]
    t = cam_pos.unsqueeze(0)                         # [1,3]
    K = torch.tensor([[max(h,w), 0, w/2],
                      [0, max(h,w), h/2],
                      [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)
    return {"K": K, "R": R, "t": t, "image_shape": (h, w)}

# 3. 渲染循环
frames = []
# 给模型挂一个最简单的 step_tracker
class MockStepTracker:
    def get_step(self):
        return 0
with torch.no_grad():
    for i in range(60):
        camera = build_camera(i * 6)
        # 极简 batch，只给 context
        # batch = {
        #     "context": {
        #         "image": torch.zeros(1, 1, 3, 176, 176, device=device),
        #         "camera": camera,
        #     }
        # }
        # 1. 先构造相机内参、外参
        # 1. 构造相机参数
        # 1. 视,图数改成 2
        # B, N_CTX, N_TGT = 1, 2, 1
        N_CTX, N_TGT = 2, 1
        H, W = 88, 88
        fx = fy = max(H, W)
        cx, cy = W / 2, H / 2

        # 1. 基础张量
        K   = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=torch.float32, device=device)
        extr= torch.eye(4, device=device)
        near= torch.tensor([0.1], device=device)
        far = torch.tensor([10.0], device=device)

        # 2. context 2 视图
        K_ctx   = K.unsqueeze(0).repeat(N_CTX,1,1)
        extr_ctx= extr.unsqueeze(0).repeat(N_CTX,1,1)
        near_ctx= near.repeat(N_CTX)
        far_ctx = far.repeat(N_CTX)
        # rgb_ctx = torch.zeros(N_CTX, 3, H, W, device=device)
        from PIL import Image
        import numpy as np

        img = Image.open("demo.png").convert("RGB").resize((W, H))
        rgb_ctx = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.
        rgb_ctx = rgb_ctx.repeat(N_CTX, 1, 1, 1).to(device)

        context_views = {
            "image":      rgb_ctx.unsqueeze(0),
            "extrinsics": extr_ctx.unsqueeze(0),
            "intrinsics": K_ctx.unsqueeze(0),
            "near":       near_ctx.unsqueeze(0),
            "far":        far_ctx.unsqueeze(0),
            "index":      torch.arange(N_CTX, device=device).unsqueeze(0),
        }

        dtarget_extr = extr_ctx[:1].unsqueeze(0)  # [1,1,4,4]
        target_K    = K_ctx[:1].unsqueeze(0)     # [1,1,3,3]
        target_near = near_ctx[:1].unsqueeze(0)  # [1,1]
        target_far  = far_ctx[:1].unsqueeze(0)   # [1,1]

        target_views = {
            "image":      rgb_ctx[:1].unsqueeze(0),          # [1,1,3,H,W]
            "extrinsics": extr_ctx[:1].unsqueeze(0),         # [1,1,4,4]
            "intrinsics": K_ctx[:1].unsqueeze(0),            # [1,1,3,3]
            "near":       near_ctx[:1].unsqueeze(0),         # [1,1]
            "far":        far_ctx[:1].unsqueeze(0),          # [1,1]
            "index":      torch.arange(N_TGT, device=device).unsqueeze(0),  # [1,1]
            "camera": {
                "K":           K_ctx[:1].unsqueeze(0),
                "R":           extr_ctx[:1].unsqueeze(0)[..., :3, :3],
                "t":           extr_ctx[:1].unsqueeze(0)[..., :3, 3],
                "image_shape": (H, W),
                "near":        near_ctx[:1].unsqueeze(0),
                "far":         far_ctx[:1].unsqueeze(0),
            },
        }

        # 4. 拼 batch
        batch = {
            "context": context_views,
            "target":  target_views,
        }
        print("context.image shape:", batch["context"]["image"].shape)
        # 5. 调试
        print("batch keys:", batch.keys())
        print("context keys:", batch["context"].keys())

        # 3. 推理
        with torch.no_grad():
            # gaussians = model.encoder(batch, 0)
            gaussians = model.encoder(batch["context"], 0)
            bg_color = torch.tensor(cfg.dataset.background_color, device=device)
        out = model.decoder(
            gaussians,
            batch["target"]["extrinsics"],   # [1,1,4,4]
            batch["target"]["intrinsics"],   # [1,1,3,3]
            batch["target"]["near"],         # [1,1]
            batch["target"]["far"],          # [1,1]
            batch["target"]["camera"]["image_shape"],  # (H,W)
        )
        pred_image = out.color  # [1, 1, 3, H, W]
        img = pred_image[0, 0].clamp(0, 1)  # [3, H, W]
        # 2. 打包成统一格式
        # out = {"pred_image": pred_image}
        # img = out["pred_image"][0].clamp(0, 1)          # [3,H,W]
        img = (img * 255).byte().cpu().numpy().transpose(1, 2, 0)
        frames.append(img)

        # 4. 写 GIF
        gif_path = ckpt_path.parent / "quick.gif"
        imageio.mimsave(gif_path, frames, fps=12)
        print("GIF 已保存 →", gif_path)