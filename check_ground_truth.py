import json

# 填入你清理好的 json 文件路径
json_path = "dataset_llava_format_10000.json"
# 填入你刚才测试的 CT 图像文件名
target_image = "1000705940001/CT175340_2506621215_5.1_Routine_Chest_0.8_sec.7.5mm.nii.gz"

print(f"[*] 正在读取 {json_path} ...")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

found = False
for item in data:
    # 检查 image 字段是否匹配
    if item.get('image') == target_image:
        print("\n" + "="*50)
        print(f"[找到了！原始报告 Ground Truth]:")
        # 遍历对话，把原本医生写的报告打印出来
        for conv in item['conversations']:
            if conv['from'] == 'gpt': # 'gpt' 或者 'assistant'，取决于你的 json 格式
                print(conv['value'])
        print("="*50 + "\n")
        found = True
        break

if not found:
    print(f"[!] 没有在 JSON 中找到关于 {target_image} 的记录。")