import pandas as pd
import json
import os
import random

csv_path = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/train_reports.csv"
output_json = "dataset_llava_format_10000.json"

# 1. 读取 CSV 时，直接使用 fillna('') 把所有空单元格替换为空字符串
df = pd.read_csv(csv_path, encoding='utf-8').fillna('')
json_data = []


human_prompts = [
    "<image>\n请详细描述该CT的影像学特征，并给出诊断意见。",
    "<image>\n作为专业的放射科医生，请解读这份3D CT影像，提供影像所见及结论。",
    "<image>\n请分析这组CT扫描序列，生成一份结构化的医疗影像报告。",
    "<image>\n基于提供的CT影像数据，请说明扫描范围内的解剖结构变化及异常诊断。",
    "<image>\n请查阅该CT影像，描述其中的阳性与阴性发现，并给出最终的影像所得。"
]

for idx, row in df.iterrows():
    volume_name = str(row.get('VolumeName', '')).strip()
    if not volume_name.endswith('.nii.gz'):
        continue

    # 提取病人ID文件夹名 (SourceFolder)
    source_folder = str(row.get('SourceFolder', '')).strip()

    # 拼接路径
    if source_folder and source_folder.lower() != 'nan':
        image_path = f"{source_folder}/{volume_name}"
    else:
        image_path = volume_name

    # 2. 提取文本内容
    findings = str(row.get('影像所见', '')).strip()
    conclusion = str(row.get('影像所得', '')).strip()

    # 3. 拦截器：如果所见和所得全是空的，直接丢弃这条数据
    if not findings and not conclusion:
        continue

    # 4. 组装符合 Qwen 逻辑的回复文本 (加入通用 think 过程)
    gpt_response = (
        "<think>\n"
        "提取3D CT影像特征，分析情况，综合评估病变。\n"
        "</think>\n\n"
    )
    if findings:
        gpt_response += f"影像所见：{findings}\n"
    if conclusion:
        gpt_response += f"影像所得：{conclusion}"

    # 构造 LLaVA 标准对话格式
    item = {
        "id": volume_name.replace('.nii.gz', ''),
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": random.choice(human_prompts) # 每次随机抽取一种问法
            },
            {
                "from": "gpt",
                "value": gpt_response.strip()
            }
        ]
    }
    json_data.append(item)

# 保存为 JSON 文件
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成！经过清洗，共生成 {len(json_data)} 条数据，已保存至 {output_json}")