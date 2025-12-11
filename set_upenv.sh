#!/bin/bash
set -e
# ========== 用户仅需修改的两处 ==========
PYTHON_VER=3.10.18          # 你实测通过的版本
CUDA_ARCH="8.0"             # A100 对应计算能力
# ========================================

echo "=== 1. 创建 conda 环境 ==="
conda create -n venv310 python="$PYTHON_VER" -y
conda activate venv310

echo "=== 2. PyTorch 11.8 套装 ==="
pip install --force-reinstall --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    -f https://download.pytorch.org/whl/cu118

echo "=== 3. 其余 PyPI 依赖 ==="
# 先降级 numpy 避免编译冲突
pip install "numpy<2" --force-reinstall --no-cache-dir
# 一次性安装项目 requirements（不含 rasterizer）
pip install -r requirements.txt

echo "=== 4. 编译 latent-gaussian-rasterization ==="
cd latent-gaussian-rasterization-main
# （glm 已内置在 third_party/glm/，无需 git）
TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" pip install --no-cache-dir .
cd ..

echo "=== 5. 验证 ==="
python -c "import diff_gaussian_rasterization, torch; print('✅ 安装成功'); print('CUDA:', torch.cuda.is_available())"

echo "=== 6. 训练示例 ==="
echo "运行：  python -m src.main +experiment=co3d_hydrant"