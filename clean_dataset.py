import json
import os
import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm

# ================= 配置区 =================
INPUT_JSON = "finetune_data_thinking.json"
OUTPUT_JSON = "finetune_data_clean.json"
IMAGE_FOLDER = "../CT-CLIP-main/dataset/pretrain_processed_train_data"


# ==========================================

def check_nii_validity(nii_path):
    """
    不仅检查文件可读性，还要模拟 train.py 的重采样计算，确保维度不会坍塌为 0
    """
    try:
        # 1. 强力过滤定位片 (Scout)
        if "scout" in nii_path.lower():
            return False, "定位片 (Scout) 已跳过"

        # 2. 尝试加载
        img = nib.load(nii_path)
        img_data = img.get_fdata()

        # 3. 维度检查
        # 正常 3D NIfTI 在 nibabel 中通常是 (W, H, D)
        if len(img_data.shape) != 3:
            return False, f"维度异常: {img_data.shape}"

        # 4. 模拟重采样计算逻辑 (同步 train.py)
        zooms = img.header.get_zooms()
        # train.py 顺序: current_spacing = (zooms[2], zooms[0], zooms[1]) -> (Z, X, Y)
        # target_spacing = (1.5, 0.75, 0.75)
        d, w, h = img_data.shape[2], img_data.shape[0], img_data.shape[1]

        new_d = int(d * (zooms[2] / 1.5))
        new_w = int(w * (zooms[0] / 0.75))
        new_h = int(h * (zooms[1] / 0.75))

        if new_d <= 0:
            return False, f"深度重采样后归零 (原深度:{d}, 间距:{zooms[2]:.2f})"
        if new_w <= 0 or new_h <= 0:
            return False, f"宽/高重采样后归零"

        # 5. 检查 NaN
        if np.isnan(img_data).any():
            return False, "包含 NaN"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def main():
    if not os.path.exists(INPUT_JSON):
        print(f"错误: 找不到输入文件 {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"开始清洗数据集，原始总数: {len(data)}")

    clean_data = []
    bad_count = 0
    error_summary = {}

    for item in tqdm(data):
        img_relative_path = item["image"]
        img_absolute_path = os.path.join(IMAGE_FOLDER, img_relative_path)

        if not os.path.exists(img_absolute_path):
            bad_count += 1
            error_summary["文件缺失"] = error_summary.get("文件缺失", 0) + 1
            continue

        is_valid, reason = check_nii_validity(img_absolute_path)

        if is_valid:
            clean_data.append(item)
        else:
            bad_count += 1
            # 分类统计错误
            cat = reason.split(":")[0]
            error_summary[cat] = error_summary.get(cat, 0) + 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 40)
    print(f"清洗完成！保留: {len(clean_data)} 条 | 剔除: {bad_count} 条")
    print("-" * 40)
    for msg, count in error_summary.items():
        print(f" - {msg}: {count} 个")
    print("=" * 40)


if __name__ == "__main__":
    main()