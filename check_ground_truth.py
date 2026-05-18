import json

# 填入你清理好的 json 文件路径
json_path = "dataset_llava_format_10000.json"
# 填入你刚才测试的 CT 图像文件名
target_image = "1000705940001/CT175340_2506621215_5.1_Routine_Chest_0.8_sec.7.5mm.nii.gz"

print(f"[*] 正在读取 {json_path} ...")
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    found = False
    for item in data:
        # 检查 image 字段是否匹配
        if item.get('image') == target_image:
            print("\n" + "=" * 60)
            print(f"🎯 [成功找到！以下是该图片对应的完整 JSON 数据]:\n")

            # 使用 json.dumps 配合 indent=4，将整个对象格式化打印出来，保持中文不乱码
            print(json.dumps(item, indent=4, ensure_ascii=False))

            print("\n" + "=" * 60)
            found = True
            break

    if not found:
        print(f"\n[!] 警告：没有在 {json_path} 中找到包含 {target_image} 的记录。")
        print("请检查文件名是否完全一致，或者该文件是否在清理脏数据时被剔除了。")

except FileNotFoundError:
    print(f"[!] 找不到文件: {json_path}，请确认路径是否正确。")