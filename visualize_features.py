import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import os
# 转换特征图npy到特征图rgb
# 特征图目录
feature_dir = Path("outputs/test/feature_maps")
output_dir = Path("outputs/test/feature_visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

# 获取所有特征图文件
npy_files = list(feature_dir.glob("*_feat.npy"))
print(f"找到 {len(npy_files)} 个特征图文件")

for npy_file in npy_files:
    print(f"处理: {npy_file.name}")
    
    # 加载特征图 [C, H, W]
    feature_map = np.load(npy_file)
    C, H, W = feature_map.shape
    
    # 展平为 [H*W, C]
    features_flat = feature_map.reshape(C, -1).T  # [H*W, C]
    
    # PCA 降到 3 维
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_flat)  # [H*W, 3]
    
    # 归一化到 [0, 1]
    features_pca = (features_pca - features_pca.min(axis=0)) / (features_pca.max(axis=0) - features_pca.min(axis=0) + 1e-8)
    
    # 变回 [H, W, 3]
    vis_img = features_pca.reshape(H, W, 3)
    
    # 保存可视化结果
    output_path = output_dir / npy_file.name.replace("_feat.npy", "_pca.png")
    plt.imsave(output_path, vis_img)
    
    # 可选：保存解释方差比例
    explained_var = pca.explained_variance_ratio_
    print(f"  前3维解释方差: {explained_var.sum():.3f}")

print("完成！")