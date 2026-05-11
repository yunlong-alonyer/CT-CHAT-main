import json
import os
import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm

# ================= 配置区 =================
INPUT_JSON = "finetune_data_thinking.json"
OUTPUT_JSON = "finetune_data_clean.json"
# 注意：这里请填写你 .sh 脚本中 --image_folder 指向的绝对路径
IMAGE_FOLDER = "../CT-CLIP-main/dataset/pretrain_processed_train_data"


# ==========================================

def check_nii_validity(nii_path):
    """
    模拟训练时的读取逻辑，检查文件是否健康
    """
    try:
        # 1. 尝试使用 nibabel 加载
        img = nib.load(nii_path)
        data = img.get_fdata()

        # 2. 检查维度 (模拟 nii_img_to_tensor 中的判断)
        # 如果 D=1 (单层) 或者数据为空，通常会导致插值或卷积失败
        if data.shape[-1] < 2 or np.isnan(data).any():
            return False, "维度异常或包含NaN"

        # 3. 模拟数据类型检查
        tensor_data = torch.from_numpy(data).float()

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

        # 检查文件是否存在
        if not os.path.exists(img_absolute_path):
            bad_count += 1
            error_msg = "文件不存在"
            error_summary[error_msg] = error_summary.get(error_msg, 0) + 1
            continue

        # 检查文件内容是否可读
        is_valid, reason = check_nii_validity(img_absolute_path)

        if is_valid:
            clean_data.append(item)
        else:
            bad_count += 1
            error_summary[reason] = error_summary.get(reason, 0) + 1
            # print(f"剔除坏数据: {img_relative_path} | 原因: {reason}")

    # 保存结果
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 30)
    print(f"清洗完成！")
    print(f"保留有效数据: {len(clean_data)} 条")
    print(f"剔除坏数据: {bad_count} 条")
    print("-" * 30)
    print("错误类型统计:")
    for msg, count in error_summary.items():
        print(f" - {msg}: {count} 个")
    print(f"结果已保存至: {OUTPUT_JSON}")
    print("=" * 30)


if __name__ == "__main__":
    main()