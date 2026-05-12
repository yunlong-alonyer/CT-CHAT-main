import json

# 1. 这里填入你最原始、未被污染的数据集文件
input_file = "dataset_llava_format.json"
output_file = "finetune_data_qwen.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
for item in data:
    # 必须要有图像，如果没有图像则跳过（防止因为空图像引发错误）
    if "image" not in item:
        continue

    new_item = {
        "id": item.get("id", ""),
        "image": item["image"],
        "conversations": []
    }

    first_human_found = False

    for conv in item["conversations"]:
        role = conv["from"].lower()
        val = conv["value"]

        # 核心修复 1：彻底清除所有历史的图像占位符，防止 num_images 越界报错
        val = val.replace("<image>", "").replace("<Image>", "").replace("</Image>", "").strip()

        # 核心修复 2：针对人类的话语，只在第一次出现时在其头部加上唯一的 <image> 标签
        if role == "human":
            if not first_human_found:
                val = "<image>\n" + val
                first_human_found = True
            new_conv = {"from": "human", "value": val}

        # 核心修复 3：针对模型的回答，注入 Qwen 必须的 <think> 思考链标签
        elif role == "gpt":
            # 确保不重复添加 think 标签
            if "<think>" not in val:
                val = f"<think>\n结合3D CT影像特征，依次分析肺部纹理、实质密度、胸膜及纵隔情况，综合评估病变性质。\n</think>\n\n{val}"
            new_conv = {"from": "gpt", "value": val}

        else:
            new_conv = {"from": role, "value": val}

        new_item["conversations"].append(new_conv)

    new_data.append(new_item)

# 2. 保存为最终微调文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"✅ 处理完成！共清洗了 {len(new_data)} 条有效数据。")
print(f"✅ 最终文件已保存至: {output_file}")
print("👉 接下来请将 finetune_qwen.sh 中的 --data_path 修改为该文件路径！")