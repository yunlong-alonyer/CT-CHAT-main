import json
import os


def convert_format(input_json_path, output_json_path):
    # 1. 加载原始 JSON 数据
    if not os.path.exists(input_json_path):
        print(f"错误: 找不到文件 {input_json_path}")
        return

    with open(input_json_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    new_data = []

    for item in old_data:
        # --- 处理图像路径 ---
        old_image_path = item.get("image", "")

        # 将 .nii.gz 替换为 .png
        if old_image_path.endswith(".nii.gz"):
            new_image_path = old_image_path.replace(".nii.gz", ".png")
        elif old_image_path.endswith(".nii"):
            new_image_path = old_image_path.replace(".nii", ".png")
        else:
            new_image_path = old_image_path

        # --- 初始化新的格式结构 ---
        new_item = {
            "messages": [],
            "images": [new_image_path]
        }

        # --- 处理对话消息 ---
        for conv in item.get("conversations", []):
            # 确定角色
            old_role = conv.get("from")
            if old_role == "human":
                new_role = "user"
            elif old_role == "gpt":
                new_role = "assistant"
            else:
                new_role = old_role

            # 确保 content 被定义，即使 value 为空
            raw_value = conv.get("value", "")

            content_value = raw_value.replace("<image>\n", "<image>\n")

            # 添加到 messages (明确指定 role 和 content)
            new_item["messages"].append({
                "role": new_role,
                "content": content_value
            })

        new_data.append(new_item)

    # 2. 保存为新的 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成！已处理 {len(new_data)} 条数据，保存至 {output_json_path}")


if __name__ == "__main__":
    INPUT_JSON = "dataset_llava_format_10000_nothink.json"
    OUTPUT_JSON = "dataset_new_format.json"

    convert_format(INPUT_JSON, OUTPUT_JSON)