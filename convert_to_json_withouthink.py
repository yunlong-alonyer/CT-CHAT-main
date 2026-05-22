import pandas as pd
import json
import os
import random

csv_path = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/train_reports.csv"
output_json = "dataset_llava_format_10000_nothink.json"

# 1. 读取 CSV，fillna('') 确保空值处理
df = pd.read_csv(csv_path, encoding='utf-8').fillna('')
json_data = []

# 多样化 Prompt，保持原样
human_prompts = [
    "<image>\n请详细描述该CT的影像学特征，并给出诊断意见。",
    "<image>\n作为专业的放射科医生，请解读这份3D CT影像，提供影像所见及影像所得。",
    "<image>\n请分析这组CT扫描序列，生成一份结构化的医疗影像报告。",
    "<image>\n基于提供的CT影像数据，请说明扫描范围内的解剖结构变化及异常诊断。",
    "<image>\n请查阅该CT影像，描述其中的阳性与阴性发现，并给出最终的影像所得。",
    "<image>\n你能为以下CT图像生成报告吗？",
    "<image>\n请提供所提及的CT图像的放射学报告。",
    "<image>\n我需要给定CT图像的放射学报告。",
    "<image>\n你能为这张CT扫描创建一份报告吗？",
    "<image>\n你介意为指定的CT图像生成放射学报告吗？",
    "<image>\n请为所提供的CT图像生成报告。",
    "<image>\n你能为所附的CT图像出具放射学报告吗？",
    "<image>\n我需要给定CT图像的详细报告。",
    "<image>\n你能为这张CT扫描撰写放射学报告吗？",
    "<image>\n请给出指定CT图像的放射学报告。",
    "<image>\n为该CT生成放射学报告。",
    "<image>\n为这张CT图像出具报告。",
    "<image>\n为以下CT扫描撰写放射学报告。",
    "<image>\n为这张CT创建报告。",
    "<image>\n提供这张CT图像的放射学报告。",
    "<image>\n你能为以下CT图像生成报告吗？",
    "<image>\n请提供所提及的CT图像的放射学报告。",
    "<image>\n我需要给定CT图像的放射学报告。",
    "<image>\n你能为这个CT图像创建一份报告吗？",
    "<image>\n你介意为指定的CT图像生成放射学报告吗？",
    "<image>\n请为所提供的CT图像生成报告。",
    "<image>\n你能为所附的CT图像出具放射学报告吗？",
    "<image>\n我需要给定CT图像的详细报告。",
    "<image>\n你能为这个CT图像撰写放射学报告吗？",
    "<image>\n请给出指定CT图像的放射学报告。",
    "<image>\n为该CT图像生成放射学报告。",
    "<image>\n为这个CT图像出具报告。",
    "<image>\n为以下CT图像撰写放射学报告。",
    "<image>\n为这个CT图像创建报告。",
    "<image>\n提供这个CT图像的放射学报告。",
    "<image>\n你能为以下CT扫描生成报告吗？",
    "<image>\n请提供所提及的CT扫描的放射学报告。",
    "<image>\n我需要给定CT扫描的放射学报告。",
    "<image>\n你能为这张CT扫描创建一份报告吗？",
    "<image>\n你介意为指定的CT扫描生成放射学报告吗？",
    "<image>\n请为所提供的CT扫描生成报告。",
    "<image>\n你能为所附的CT扫描出具放射学报告吗？",
    "<image>\n我需要给定CT扫描的详细报告。",
    "<image>\n你能为这张CT扫描撰写放射学报告吗？",
    "<image>\n请给出指定CT扫描的放射学报告。",
    "<image>\n为该CT扫描生成放射学报告。",
    "<image>\n为这张CT扫描出具报告。",
    "<image>\n为以下CT扫描撰写放射学报告。",
    "<image>\n为这张CT扫描创建报告。",
    "<image>\n提供这张CT扫描的放射学报告。",
]

for idx, row in df.iterrows():
    volume_name = str(row.get('VolumeName', '')).strip()
    if not volume_name.endswith('.nii.gz'):
        continue

    # 提取病人ID文件夹名
    source_folder = str(row.get('SourceFolder', '')).strip()

    # 拼接路径
    if source_folder and source_folder.lower() != 'nan':
        image_path = f"{source_folder}/{volume_name}"
    else:
        image_path = volume_name

    # 2. 提取文本内容
    findings = str(row.get('影像所见', '')).strip()
    conclusion = str(row.get('影像所得', '')).strip()

    # 3. 拦截器：确保有内容
    if not findings and not conclusion:
        continue

    # 4. 直接组装报告文本 (彻底移除 think 标签)
    report_parts = []
    if findings:
        report_parts.append(f"影像所见：{findings}")
    if conclusion:
        report_parts.append(f"影像所得：{conclusion}")

    gpt_response = "\n\n".join(report_parts)

    # 构造 LLaVA 标准对话格式
    item = {
        "id": volume_name.replace('.nii.gz', ''),
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": random.choice(human_prompts)
            },
            {
                "from": "gpt",
                "value": gpt_response
            }
        ]
    }
    json_data.append(item)

# 保存为 JSON 文件
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成！已剔除 <think> 标签，共生成 {len(json_data)} 条数据，已保存至 {output_json}")