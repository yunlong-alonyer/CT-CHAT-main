import json
import os
import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm

# ================= 配置区 =================
# 输入你原始的、包含问题的 JSON 文件
INPUT_JSON = "finetune_data_thinking.json"
# 输出清洗后、可直接用于训练的 JSON 文件
OUTPUT_JSON = "finetune_data_clean.json"
# 请务必填写与 finetune_qwen.sh 中 --image_folder 一致的路径
IMAGE_FOLDER = "../CT-CLIP-main/dataset/pretrain_processed_train_data"
# ==========================================

def check_nii_validity(nii_path):
    """
    同步 train.py 中的 nii_img_to_tensor 逻辑，
    捕获导致 'axes don't match array' 的异常文件。
    """
    try:
        # 1. 尝试加载 NIfTI 文件
        nii_img = nib.load(str(nii_path))
        img_data = nii_img.get_fdata()

        # 2. 检查维度 (核心检查点)
        # 报错 'axes don't match' 通常是因为 transpose(2, 0, 1) 要求数组必须是 3 维
        # 如果文件是 2 维或 4 维 (例如带通道或时间轴)，这里会直接报错
        try:
            _ = img_data.transpose(2, 0, 1)
        except ValueError:
            return False, f"维度不匹配 (shape={img_data.shape}, 无法执行 transpose(2,0,1))"

        # 3. 检查是否存在 NaN (避免训练 Loss 变成 NaN)
        if np.isnan(img_data).any():
            return False, "文件包含 NaN 值"

        # 4. 模拟重采样前的基本属性检查
        zooms = nii_img.header.get_zooms()
        if len(zooms) < 3:
            return False, f"Header 信息缺失 (zooms={zooms})"

        return True, "OK"
    except Exception as e:
        return False, str(e)

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"错误: 找不到输入文件 {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"开始清洗数据集，原始样本总数: {len(data)}")

    clean_data = []
    bad_count = 0
    error_summary = {}

    for item in tqdm(data):
        # 获取图像路径
        img_relative_path = item.get("image", "")
        if not img_relative_path:
            bad_count += 1
            error_msg = "JSON 条目缺失 image 字段"
            error_summary[error_msg] = error_summary.get(error_msg, 0) + 1
            continue

        img_absolute_path = os.path.join(IMAGE_FOLDER, img_relative_path)

        # 1. 检查物理文件是否存在
        if not os.path.exists(img_absolute_path):
            bad_count += 1
            error_msg = "物理文件不存在"
            error_summary[error_msg] = error_summary.get(error_msg, 0) + 1
            continue

        # 2. 检查文件内部结构是否合规
        is_valid, reason = check_nii_validity(img_absolute_path)

        if is_valid:
            clean_data.append(item)
        else:
            bad_count += 1
            # 简化错误分类以便统计
            category = reason if "维度" in reason else ("读取失败: " + reason[:30])
            error_summary[category] = error_summary.get(category, 0) + 1

    # 保存清洗后的 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"清洗完成！统计结果如下：")
    print(f" - 保留有效数据: {len(clean_data)} 条")
    print(f" - 剔除问题数据: {bad_count} 条")
    print("-" * 50)
    print("错误类型细分:")
    for msg, count in error_summary.items():
        print(f"  * {msg}: {count} 个")
    print("-" * 50)
    print(f"结果已保存至: {OUTPUT_JSON}")
    print("请修改 finetune_qwen.sh 中的 --data_path 为此文件路径后重新开始训练。")
    print("=" * 50)

if __name__ == "__main__":
    main()