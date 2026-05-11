import json
import os

# 指向你当前报错的 JSON 文件
JSON_PATH = "finetune_data_clean.json"


def main():
    if not os.path.exists(JSON_PATH):
        print(f"找不到文件 {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_count = 0

    for item in data:
        if "image" not in item:
            continue

        has_added_image_token = False

        for conv in item["conversations"]:
            # 1. 暴力清除当前轮次中所有的图像占位符及各种可能变体
            text = conv["value"]
            text = text.replace("<image>", "").replace("<Image>", "").replace("</Image>", "")
            text = text.replace("<image>\n", "").replace("\n<image>", "").strip()

            # 2. 确保整个多轮对话，只在第一次 human 发言的最前面加且仅加一次 <image>
            if conv["from"] == "human" and not has_added_image_token:
                text = "<image>\n" + text
                has_added_image_token = True

            conv["value"] = text

        fixed_count += 1

    # 直接覆盖原文件，省去修改 .sh 脚本的麻烦
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"文本 <image> 标签修复完成！共修正了 {fixed_count} 条数据。")
    print(f"已覆盖原文件：{JSON_PATH}。你现在可以直接重新启动训练！")
    print("=" * 50)


if __name__ == "__main__":
    main()