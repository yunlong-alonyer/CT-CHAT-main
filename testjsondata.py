import json

with open("finetune_data_clean.json", "r") as f:
    data = json.load(f)

for i, item in enumerate(data):
    all_text = "".join([c["value"] for c in item["conversations"]])
    count = all_text.count("<image>")
    if count > 1:
        print(f"条目 {i} 包含 {count} 个 <image> 标签，这会导致崩溃！图像路径: {item.get('image')}")