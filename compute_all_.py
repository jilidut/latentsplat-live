import torch
import torch_fidelity  # 导入 torch_fidelity
import subprocess
from pathlib import Path
from PIL import Image
from torchvision import transforms
import piq
import numpy as np

pred_dir = "outputs/test/re10k_extra"
gt_dir = "outputs/test/gt_images_extra"

# pred_dir = "outputs/test/re10k_intra"
# gt_dir = "outputs/test/gt_images_intra"

# pred_dir = "outputs/test/acid"
# gt_dir = "outputs/test/gt_images_acid"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pred_batch(start_idx, batch_size):
    """加载预测图（有 color 子文件夹）"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    all_files = sorted(Path(pred_dir).glob("re10k/*/*/*.png"))
    batch_files = all_files[start_idx:start_idx + batch_size]
    
    images = []
    for fname in batch_files:
        img = Image.open(fname).convert('RGB')
        img = transform(img).unsqueeze(0)
        images.append(img)
    
    if not images:
        return None
    return torch.cat(images, dim=0).to(device)

def load_gt_batch(start_idx, batch_size):
    """加载 GT 图（没有 color，直接在场景/索引下）"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    all_files = sorted(Path(gt_dir).glob("*/*/*.png"))
    batch_files = all_files[start_idx:start_idx + batch_size]
    
    images = []
    for fname in batch_files:
        img = Image.open(fname).convert('RGB')
        img = transform(img).unsqueeze(0)
        images.append(img)
    
    if not images:
        return None
    return torch.cat(images, dim=0).to(device)

# 分批计算 DISTS
print("计算 DISTS...")
batch_size = 32
dists_values = []

all_files = sorted(Path(pred_dir).glob("re10k/*/*/*.png"))
num_images = len(all_files)
print(f"总图片数: {num_images}")

for i in range(0, num_images, batch_size):
    print(f"处理批次 {i//batch_size + 1}/{(num_images + batch_size - 1)//batch_size}")
    
    pred_batch = load_pred_batch(i, batch_size)
    gt_batch = load_gt_batch(i, batch_size)
    
    if pred_batch is None or gt_batch is None:
        continue
    
    with torch.no_grad():
        batch_dists = piq.DISTS()(pred_batch, gt_batch).item()
    dists_values.append(batch_dists * len(pred_batch))

# 计算加权平均
dists = sum(dists_values) / num_images
print(f"DISTS: {dists:.6f}")

# ===== 用 torch_fidelity API 计算 FID/KID =====
print("\n计算 FID 和 KID...")
try:
    metrics = torch_fidelity.calculate_metrics(
        input1=pred_dir,
        input2=gt_dir,
        cuda=True,
        fid=True,
        kid=True,
        batch_size=32,
        samples_find_deep=True,  # 递归搜索子目录
    )
    print(f"FID: {metrics['frechet_inception_distance']:.4f}")
    print(f"KID: {metrics['kernel_inception_distance_mean']:.4f}")
except Exception as e:
    print(f"FID/KID 计算失败: {e}")
    # 如果 API 失败，可以尝试用命令行
    print("\n尝试用命令行计算...")
    cmd = f"python -m torch_fidelity --gpu 0 --fid --kid --input1 {pred_dir} --input2 {gt_dir}"
    subprocess.run(cmd, shell=True)