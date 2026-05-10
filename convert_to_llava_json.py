import pandas as pd
import json
import os

csv_path = "../CT-CLIP-main/CT_CLIP/dataset/train_reports.csv"
output_json = "dataset_llava_format.json"

# 读取 CSV 文件
df = pd.read_csv(csv_path, encoding='utf-8')
json_data = []

for idx, row in df.iterrows():
    volume_name = str(row.get('VolumeName', ''))
    if not volume_name.endswith('.nii.gz'):
        continue

    # 提取病人ID文件夹名 (SourceFolder)
    source_folder = str(row.get('SourceFolder', ''))

    # 核心修改：将文件夹名和文件名拼接在一起
    # 生成格式如: "10050302800015/P0000520759_491914164_Chest.nii.gz"
    if source_folder and source_folder != 'nan':
        image_path = f"{source_folder}/{volume_name}"
    else:
        image_path = volume_name

    findings = str(row.get('影像所见', ''))
    conclusion = str(row.get('影像所得', ''))

    # 构造 LLaVA 标准对话格式
    item = {
        "id": volume_name.replace('.nii.gz', ''),
        "image": image_path,  # <--- 使用拼接后的带文件夹的路径
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n请详细描述该胸部CT的影像学特征，并给出诊断意见。"
            },
            {
                "from": "gpt",
                "value": f"影像所见：{findings}\n影像所得：{conclusion}"
            }
        ]
    }
    json_data.append(item)

# 保存为 JSON 文件
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成，共生成 {len(json_data)} 条数据，已保存至 {output_json}")