import json

with open("dataset_llava_format.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    for conv in item["conversations"]:
        if conv["from"] == "gpt":
            # 在原有的报告前注入思考占位符
            original_report = conv["value"]
            conv["value"] = f"<think>\n结合3D CT影像特征，依次分析肺部纹理、实质密度、胸膜及纵隔情况，综合评估病变性质。\n</think>\n\n{original_report}"

with open("finetune_data_thinking.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)