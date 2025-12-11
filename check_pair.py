import json, pathlib, os

pred_root = "outputs/test/latentsplat_200k_20251111_100051"
gt_root   = "/home/gengqt/latentsplat-live/datasets/re10k_extra"
index     = "assets/dataset_splits/re10k_subset_eval.json"

idx   = json.load(open(index))
err   = []

for seq, entries in idx.items():
    if not entries: continue
    for ent in entries:
        for tgt in ent["target"]:
            pred = pathlib.Path(pred_root) / seq / f"{tgt:05d}_pred_rgb.png"
            gt   = pathlib.Path(gt_root)   / seq / "frames" / f"{tgt:05d}.jpg"
            if not pred.exists():
                err.append(f"缺 pred {seq}/{tgt:05d}")
            if not gt.exists():
                err.append(f"缺 GT   {seq}/{tgt:05d}")
            # 可选：扩展名不同
            # if pred.suffix != gt.suffix:
            #     err.append(f"后缀不同 {seq}/{tgt:05d}  pred={pred.suffix}  gt={gt.suffix}")

print("总错误数:", len(err))
if err:
    print("前 10 例:", err[:10])
else:
    print("✅ 帧级完全对齐！")
