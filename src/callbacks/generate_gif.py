# src/callbacks/generate_gif.py
import math, torch, imageio
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only

class GenerateGif:
    """
    每 N 步把当前模型权重拉出来，
    用 encoder_visualizer 渲染 360° 环形视角 GIF。
    """
    def __init__(self,
                 output_dir="outputs/gif",
                 n_frames=60,
                 fps=10,
                 dist=2.0,
                 elev=20.,
                 h=176,
                 w=176):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.n_frames = n_frames
        self.fps = fps
        self.dist, self.elev = dist, elev
        self.h, self.w = h, w

    # --------------------------------------------------
    @rank_zero_only          # 多卡时只让主进程做可视化
    def generate(self, trainer, pl_module):
        device = pl_module.device
        pl_module.eval()

        images = []
        with torch.no_grad():
            for i in range(self.n_frames):
                azim = i * (360 / self.n_frames)
                camera = self._build_camera(device, azim)

                # 1. 构造极简 batch（只给 encoder 用）
                # 构造极简 batch（与 grid batch 维度一致）
                dummy_img = torch.zeros(2, 1, 3, self.h, self.w, device=device)

                # 外参 [1, 1, 4, 4]
                extrinsics_4x4 = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
                extrinsics_4x4[:, 0, :3, :3] = camera["R"]  # [1, 3, 3]
                extrinsics_4x4[:, 0, :3, 3]  = camera["t"]  # [1, 3]

                # 内参 [1, 3, 3]  ←🔥FIX  只 unsqueeze 一次
                # intrinsics = camera["K"].unsqueeze(0)
                # 内参 [1, 3, 3]
                intrinsics = camera["K"].unsqueeze(0)  # 必须是 [3, 3] 前提

                batch = {
                    "context": {
                        "image": dummy_img,
                        "camera": camera,
                        "extrinsics": extrinsics_4x4,
                        # "intrinsics": intrinsics.unsqueeze(0),  # 🔥再补一维 → [1, 1, 3, 3]
                        "intrinsics": intrinsics,  # [1, 3, 3]
                        "near": torch.tensor([[0.1]], device=device),
                        "far":  torch.tensor([[10.0]], device=device),
                        "index": torch.tensor([[0]], device=device),
                    },
                    "target": {
                        "image": dummy_img,
                        "camera": camera,
                        "extrinsics": extrinsics_4x4,
                        "intrinsics": intrinsics.unsqueeze(0),  # 🔥同上
                        "near": torch.tensor([[0.1]], device=device),
                        "far":  torch.tensor([[10.0]], device=device),
                        "index": torch.tensor([[0]], device=device),
                    }
                }

                # 2. 前向 -> 高斯
                gaussians, _ = pl_module(batch)   # 返回 (gaussians, extra)

                # 3. visualizer 渲染
                vis = pl_module.encoder_visualizer
                pred = vis.render_panorama(gaussians,
                                           camera,
                                           (self.h, self.w))   # [1,3,H,W] 0-1
                img = pred[0].clamp(0, 1)
                img = (img * 255).byte().cpu().numpy().transpose(1, 2, 0)
                images.append(img)

        # 4. 保存 gif
        step = trainer.global_step
        gif_path = self.output_dir / f"step{step:06d}.gif"
        imageio.mimsave(gif_path, images, fps=self.fps)
        print(f"[GenerateGif] 环形视角 gif 已保存 → {gif_path}")

        pl_module.train()          # 别忘了切回训练模式

    # --------------------------------------------------
    def _build_camera(self, device, azim_deg):
        azim = math.radians(azim_deg)
        elev = math.radians(self.elev)
        cam_pos = torch.tensor([
            self.dist * math.cos(elev) * math.sin(azim),
            self.dist * math.sin(elev),
            self.dist * math.cos(elev) * math.cos(azim)
        ], device=device)
        target = torch.zeros(3, device=device)
        up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

        z = torch.nn.functional.normalize(cam_pos - target, dim=-1)
        x = torch.nn.functional.normalize(torch.cross(up, z), dim=-1)
        y = torch.cross(z, x)
        R = torch.stack([x, y, z], dim=-1)          # [3,3]
        t = cam_pos
        K = torch.tensor([[max(self.h, self.w), 0, self.w / 2],
                          [0, max(self.h, self.w), self.h / 2],
                          [0, 0, 1]], device=device, dtype=torch.float32)
        return {
            "K": K.unsqueeze(0),      # [1, 3, 3]
            "R": R.unsqueeze(0),      # [1, 3, 3]
            "t": t.unsqueeze(0),      # [1, 3]
            "image_shape": (self.h, self.w)
        }