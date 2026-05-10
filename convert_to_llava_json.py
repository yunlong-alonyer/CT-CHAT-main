import pandas as pd
import json

csv_path = "../CT-CLIP-main/CT_CLIP/dataset/train_reports.csv"
output_json = "dataset_llava_format.json"

df = pd.read_csv(csv_path, encoding='utf-8')
json_data = []

for idx, row in df.iterrows():
    volume_name = str(row.get('VolumeName', ''))
    if not volume_name.endswith('.nii.gz'):
        continue

    findings = str(row.get('影像所见', ''))
    conclusion = str(row.get('影像所得', ''))

    # 构造 LLaVA 标准对话格式
    item = {
        "id": volume_name.replace('.nii.gz', ''),
        "image": volume_name,
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

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成，共生成 {len(json_data)} 条数据，已保存至 {output_json}")
