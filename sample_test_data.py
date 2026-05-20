import json
import random
import os

# ================= 配置区 =================
# 替换为你当前文件夹下生成好的原始全量 JSON 文件名
input_json_path = "dataset_llava_format_10000.json"
# 抽取的 100 例保存的文件名
output_json_path = "test_100_samples.json"
# 随机种子，确保每次运行抽取的都是同一批（如果不需要固定可以设为 None）
random.seed(42)


# ==========================================

def main():
    if not os.path.exists(input_json_path):
        print(f"❌ 找不到文件: {input_json_path}")
        return

    # 1. 读取完整数据
    print(f"[*] 正在读取数据: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"[*] 数据集总样本数: {total_samples}")

    # 2. 随机抽取 100 例（如果总数不足 100，则全量抽取）
    sample_size = min(100, total_samples)
    sampled_data = random.sample(data, sample_size)

    # 3. 保存到新的 JSON 文件中
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证中文正常显示，indent=2 保证格式美观
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功抽取 {sample_size} 例样本！")
    print(f"✅ 测试集已保存至: {output_json_path}")


if __name__ == "__main__":
    main()