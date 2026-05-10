import os
import torch
import transformers
from llava.train.train import LazySupervisedDataset, DataArguments

def test_dataset():
    # 路径配置
    DATA_PATH = "./dataset_llava_format.json"
    IMAGE_FOLDER = "../CT-CLIP-main/dataset/pretrain_processed_train_data"
    MODEL_PATH = "/home/huali/model/Qwen3.5-9B"

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 1. 完整初始化 DataArguments，解决缺失属性问题
    data_args = DataArguments(
        data_path=DATA_PATH,
        image_folder=IMAGE_FOLDER,
        is_multimodal=True,
        mm_use_im_start_end=False,  # 显式指定，防止报错
        mm_use_im_patch_token=False
    )

    print(f"[*] 正在加载数据集...")
    dataset = LazySupervisedDataset(data_path=DATA_PATH, tokenizer=tokenizer, data_args=data_args)

    # 2. 检查第一个样本
    print(f"[*] 尝试读取第 1 条数据...")
    try:
        sample = dataset[0]
        image_tensor = sample['image']
        print("\n" + "="*30)
        print(f"✅ 成功! 图像形状: {image_tensor.shape}") # 应为 [1, 32, 240, 240]
        print(f"✅ 文本 input_ids 长度: {sample['input_ids'].shape}")
        print("="*30)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = os.getcwd()
    test_dataset()