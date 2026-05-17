import json
import os
import nibabel as nib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# === 配置你的路径 ===
data_path = './dataset_llava_format_10000.json'
image_folder = '/mnt/huali/ct_dataset_10000/pretrain_processed_train_data'
clean_data_path = './dataset_llava_format_clean.json'
bad_data_log = './bad_data_list.txt'


# ====================

# 工作进程：单独处理一个样本
def check_single_item(item_data):
    i, item = item_data
    if 'image' not in item:
        # 如果没有图像（纯文本），直接通过
        return (True, item, None)

    img_path = os.path.join(image_folder, item['image'])
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError("文件不存在")

        nii_img = nib.load(img_path)

        # ⚠️ 速度与安全的权衡：
        # 如果你只要求查“文件是否存在”和“维度大小是否合法”，可以把下一行换成: img_shape = nii_img.shape
        # 但为了 100% 确保训练时不报解压错误，这里还是保留 get_fdata()，靠多进程提速
        img_data = nii_img.get_fdata()

        if img_data.ndim < 2:
            raise ValueError(f"严重畸形数据，维度过低: {img_data.shape}")

        return (True, item, None)
    except Exception as e:
        error_msg = f"Index: {i} | Image: {item['image']} | Error: {str(e)}"
        return (False, None, error_msg)


def check_dataset_fast():
    print(f"[*] 正在加载 JSON 数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clean_data = []
    bad_data_records = []

    # 包装数据以便传给多进程
    items_to_check = [(i, item) for i, item in enumerate(data)]

    # 🚀 核心加速：使用 16 个甚至 32 个进程同时读取
    # 你可以根据你服务器的 CPU 核心数 (htop 查看) 调大 max_workers
    MAX_WORKERS = 64

    print(f"[*] 开始多进程扫雷检测 (启动 {MAX_WORKERS} 个进程同时跑)...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(check_single_item, item) for item in items_to_check]

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures)):
            is_clean, item, error_msg = future.result()
            if is_clean:
                clean_data.append(item)
            else:
                bad_data_records.append(error_msg)

    # === 保存清洗后的结果 ===
    with open(clean_data_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)

    if bad_data_records:
        with open(bad_data_log, 'w', encoding='utf-8') as f:
            f.write("\n".join(bad_data_records))

    print("\n" + "=" * 40)
    print("✨ 清洗完成！")
    print(f"- 原始数据总数: {len(data)}")
    print(f"- 完好数据总数: {len(clean_data)}")
    print(f"- 损坏数据总数: {len(bad_data_records)} (详情见 {bad_data_log})")
    print("=" * 40)


if __name__ == "__main__":
    check_dataset_fast()