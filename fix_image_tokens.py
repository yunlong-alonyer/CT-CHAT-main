import json
import os

INPUT_JSON = "finetune_data_clean.json"
OUTPUT_JSON = "finetune_data_clean.json"


def main():
    if not os.path.exists(INPUT_JSON):
        print(f"错误: 找不到文件 {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_count = 0

    for item in data:
        # 如果没有图像，直接跳过 (通常多模态微调都要有图像)
        if "image" not in item:
            continue

        # 1. 暴力清除所有对话轮次中原有的图像标签
        for conv in item["conversations"]:
            # 清除可能出现的各种变体
            text = conv["value"]
            text = text.replace("<image>", "").replace("<Image>", "").replace("</Image>", "")
            text = text.replace("<image>\n", "").replace("\n<image>", "")
            conv["value"] = text.strip()

        # 2. 强制在 "human" 的第一句话最前面加上唯一的一个 <image>\n
        for conv in item["conversations"]:
            if conv["from"] == "human":
                conv["value"] = "<image>\n" + conv["value"]
                break  # 确保只加一次

        fixed_count += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("=" * 40)
    print(f"文本标签修复完成！共处理: {fixed_count} 条数据")
    print(f"新文件已保存至: {OUTPUT_JSON}")
    print("=" * 40)


if __name__ == "__main__":
    main()