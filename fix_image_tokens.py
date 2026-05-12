import json

# 1. 加载你当前的清洗后的数据
with open("finetune_data_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "image" not in item: continue

    first_human_found = False
    for conv in item["conversations"]:
        # 移除该轮对话中所有的图片标签变体
        val = conv["value"]
        val = val.replace("<image>", "").replace("<Image>", "").replace("</Image>", "").strip()

        # 仅在第一个 Human 话语前添加唯一的标签
        if conv["from"] == "human" and not first_human_found:
            val = "<image>\n" + val
            first_human_found = True

        conv["value"] = val

# 2. 覆盖原文件
with open("finetune_data_clean.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("修复完成！所有样本现在都只有唯一的 <image> 标签。")