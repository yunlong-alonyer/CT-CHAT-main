import os
import torch
import transformers
from llava.train.train import LazySupervisedDataset, DataArguments
from llava.model.language_model.llava_qwen import LlavaQwenConfig


def test_dataset():
    # 1. 路径配置（请根据实际路径核对）
    DATA_PATH = "./dataset_llava_format.json"
    IMAGE_FOLDER = "../CT-CLIP-main/dataset/pretrain_processed_train_data"
    MODEL_PATH = "/home/huali/model/Qwen3.5-9B"

    print(f"[*] 正在初始化分词器: {MODEL_PATH}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True
    )

    # 2. 模拟 DataArguments
    data_args = DataArguments()
    data_args.data_path = DATA_PATH
    data_args.image_folder = IMAGE_FOLDER
    data_args.is_multimodal = True

    print(f"[*] 正在加载数据集: {DATA_PATH}")
    # 实例化数据集
    dataset = LazySupervisedDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        data_args=data_args
    )

    print(f"[*] 数据集大小: {len(dataset)}")

    # 3. 提取第一个样本
    try:
        print("[*] 正在尝试加载第一个样本 dataset[0]...")
        sample = dataset[0]

        if 'image' in sample:
            image_tensor = sample['image']
            print("\n" + "=" * 30)
            print(f"成功! 图像张量形状: {image_tensor.shape}")
            print(f"预期形状: torch.Size([1, 32, 240, 240])")
            print("=" * 30)

            # 验证形状是否完全符合要求
            expected_shape = (1, 32, 240, 240)
            if tuple(image_tensor.shape) == expected_shape:
                print("✅ 形状验证通过！你可以开始运行 pretrain_qwen.sh 了。")
            else:
                print(f"❌ 形状不匹配！当前形状为 {tuple(image_tensor.shape)}")
        else:
            print("❌ 错误：样本中不包含 'image' 键，请检查 JSON 文件或数据集加载逻辑。")

    except Exception as e:
        print(f"❌ 运行失败！捕获到异常:\n{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置 PYTHONPATH 确保能找到 llava 模块
    os.environ["PYTHONPATH"] = os.getcwd() + ":" + os.environ.get("PYTHONPATH", "")
    test_dataset()