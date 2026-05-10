import os
import glob
import json
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CTCLIP


def apply_softmax(array):
    softmax = torch.nn.Softmax(dim=0)
    return softmax(array)


def load_and_preprocess_nii(file_path):
    img = nib.load(file_path).get_fdata()

    # 维度清洗与轴转置保持不变
    while img.ndim > 3:
        img = img[..., 0]
    if img.ndim < 3:
        img = np.expand_dims(img, axis=-1)
    img = np.transpose(img, (2, 0, 1))

    # === 修改这里：严格对齐官方的 HU 值归一化 ===
    img = np.clip(img, -1000, 1000)  # 官方截断范围
    img = (img / 1000.0).astype(np.float32)  # 官方映射范围 [-1.0, 1.0]
    # ============================================

    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    depth = tensor.shape[2]
    pad_depth = (10 - depth % 10) % 10
    if pad_depth > 0:
        tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad_depth))

    tensor = F.interpolate(tensor, size=(tensor.shape[2], 480, 480), mode='trilinear', align_corners=False)
    return tensor


def main():
    # 1. 配置文件路径
    MODEL_WEIGHT_PATH = "../checkpoint/CT-CLIP_v2.pt"
    DATA_DIR = "../../CT-CLIP-main/dataset/pretrain_processed_train_data"
    OUTPUT_JSON = "batch_screening_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device}")

    # 2. 初始化模型
    print("正在初始化模型与加载权重...")
    # 定义你刚才上传的本地文件夹绝对路径
    local_bert_path = "/home/huali/code/CT-CHAT-main/scripts/BiomedVLP-CXR-BERT-specialized"

    # 使用本地路径加载
    tokenizer = BertTokenizer.from_pretrained(local_bert_path, do_lower_case=True)
    text_encoder = BertModel.from_pretrained(local_bert_path)

    image_encoder = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4, dim_head=32, heads=8
    )

    clip = CTCLIP(
        image_encoder=image_encoder, text_encoder=text_encoder,
        dim_image=294912, dim_text=768, dim_latent=512,
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False
    )

    # 手动读取权重文件
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)

    # 允许非严格加载 (strict=False 即可忽略多余的 position_ids 键)
    clip.load_state_dict(checkpoint, strict=False)
    clip.to(device)
    clip.eval()

    # 3. 诊断疾病列表
    disease_mapping = {
        'Emphysema': '肺气肿', 'Atelectasis': '肺不张', 'Consolidation': '肺实变',
        'Lung nodule': '肺结节', 'Bronchiectasis': '支气管扩张', 'Pleural effusion': '胸腔积液',
        'Pericardial effusion': '心包积液', 'Arterial wall calcification': '主动脉硬化',
        'Lung opacity': '磨玻璃影', 'Interlobular septal thickening': 'Kerley B线'
    }

    # 4. 优化：预先计算所有文本的 Tokens (避免在循环中重复计算)
    print("正在预计算文本特征...")
    text_tokens_dict = {}
    for eng_disease in disease_mapping.keys():
        text = [f"{eng_disease} is present.", f"{eng_disease} is not present."]
        text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(
            device)
        text_tokens_dict[eng_disease] = text_tokens

    # 5. 查找所有 nii.gz 文件
    # 使用 glob 的 recursive=True 可以穿透所有子文件夹找到 nii.gz 文件
    search_pattern = os.path.join(DATA_DIR, "**", "*.nii.gz")
    nii_files = glob.glob(search_pattern, recursive=True)
    print(f"共发现 {len(nii_files)} 个 CT 文件等待处理。\n")

    results_dict = {}

    # 6. 开始批量推理
    with torch.no_grad():
        for file_path in tqdm(nii_files, desc="处理进度"):
            file_name = os.path.basename(file_path)

            try:
                ct_tensor = load_and_preprocess_nii(file_path).to(device)
            except Exception as e:
                print(f"\n[错误] 文件读取失败跳过: {file_name} | 原因: {str(e)}")
                continue

            file_results = {}
            for eng_disease, zh_disease in disease_mapping.items():
                # 获取预计算的 token 并推理
                tokens = text_tokens_dict[eng_disease]
                output = clip(tokens, ct_tensor, device=device)
                output = apply_softmax(output)

                prob_present = output[0].item()
                file_results[zh_disease] = round(prob_present, 4)  # 保留4位小数

            # 使用相对路径或文件名作为 key 存入字典，方便后续对齐
            # 记录上级文件夹名（即病人ID）和文件名
            parent_folder = os.path.basename(os.path.dirname(file_path))
            unique_key = f"{parent_folder}/{file_name}"
            results_dict[unique_key] = file_results

    # 7. 保存结果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 批量推理完成！结果已保存至: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()