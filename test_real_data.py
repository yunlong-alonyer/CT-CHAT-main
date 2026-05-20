import os
import torch
import numpy as np
import pydicom
import torch.nn.functional as F
import nibabel as nib
import types
from transformers import AutoTokenizer, AutoConfig
import re

# 引入你的模型架构
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


# =========================================================================
# 1. 影像读取与预处理函数
# =========================================================================
def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [max(1, int(original_shape[i] * scaling_factors[i])) for i in range(len(original_shape))]
    return F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)


def process_nii_for_v2(nii_path):
    nii_img = nib.load(str(nii_path))
    img_data = nii_img.get_fdata()
    zooms = nii_img.header.get_zooms()
    xy_spacing, z_spacing = float(zooms[0]), float(zooms[2])

    while img_data.ndim > 3: img_data = img_data[..., 0]
    if img_data.ndim == 2: img_data = img_data[:, :, np.newaxis]

    img_data = img_data.transpose(2, 0, 1)
    target = (1.5, 0.75, 0.75)
    current = (z_spacing, xy_spacing, xy_spacing)

    tensor = torch.tensor(img_data.copy()).float().unsqueeze(0).unsqueeze(0)
    tensor = resize_array(tensor, current, target)
    img_data = np.transpose(tensor[0][0].numpy(), (1, 2, 0))

    img_data = np.clip(img_data, -1000, 1000) / 1000.0
    tensor = torch.tensor(img_data.astype(np.float32))

    target_shape = (480, 480, 40)
    h, w, d = tensor.shape
    tensor = tensor[max((h - target_shape[0]) // 2, 0):min(max((h - target_shape[0]) // 2, 0) + target_shape[0], h),
    max((w - target_shape[1]) // 2, 0):min(max((w - target_shape[1]) // 2, 0) + target_shape[1], w),
    max((d - target_shape[2]) // 2, 0):min(max((d - target_shape[2]) // 2, 0) + target_shape[2], d)]

    tensor = torch.nn.functional.pad(tensor, (
        (target_shape[2] - tensor.size(2)) // 2,
        target_shape[2] - tensor.size(2) - (target_shape[2] - tensor.size(2)) // 2,
        (target_shape[1] - tensor.size(1)) // 2,
        target_shape[1] - tensor.size(1) - (target_shape[1] - tensor.size(1)) // 2,
        (target_shape[0] - tensor.size(0)) // 2,
        target_shape[0] - tensor.size(0) - (target_shape[0] - tensor.size(0)) // 2
    ), value=-1)
    return tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).cuda().bfloat16()


# =========================================================================
# 2. 模型初始化
# =========================================================================
QWEN_DIR = "../../model/Qwen3.5-9B"
CT_CLIP_PATH = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500_full.pt"
PROJECTOR_WEIGHTS_PATH = "./checkpoints/test2/checkpoint-26/mm_projector.bin"
NII_PATH = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data/10053105940002/CT163369_1090606624_02_HeadRoutine_Seq.nii.gz"

tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
raw_config = AutoConfig.from_pretrained(QWEN_DIR, trust_remote_code=True)
setattr(raw_config, "mm_vision_tower", "ctclip");
setattr(raw_config, "mm_hidden_size", 512)
setattr(raw_config, "vision_tower_path", CT_CLIP_PATH);
setattr(raw_config, "image_token_id", 248056)

model = LlavaQwenForCausalLM.from_pretrained(QWEN_DIR, config=raw_config, torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
model.get_model().vision_tower = build_vision_tower(raw_config)
model.get_model().mm_projector = build_vision_projector(raw_config)

# 加载权重
checkpoint = torch.load(PROJECTOR_WEIGHTS_PATH, map_location="cpu")
model.get_model().mm_projector.load_state_dict({k.replace("mm_projector.", ""): v for k, v in checkpoint.items()})

model.get_model().mm_projector.to(torch.bfloat16).cuda()
model.get_model().vision_tower.to(torch.float32).cuda()
model.eval()


# 动态精度补丁
def patched_encode_images(self, images):
    features = self.get_model().get_vision_tower()(images.to(torch.float32))
    return self.get_model().mm_projector(features.to(torch.bfloat16))


model.encode_images = types.MethodType(patched_encode_images, model)

# =========================================================================
# 3. 推理逻辑
# =========================================================================
conv = conv_templates["qwen"].copy()
conv.system = "你是一个专业的医疗AI助手CT-CHAT。请直接描述影像内容。"
raw_text = f"{DEFAULT_IMAGE_TOKEN}\n这是一张CT影像，请简要告诉我你在影像中看到了什么？"
conv.append_message(conv.roles[0], raw_text)
conv.append_message(conv.roles[1], None)

input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
    0).cuda()
images = process_nii_for_v2(NII_PATH)

print("\n[Debug] 正在拼装特征...")
with torch.no_grad():
    _, _, _attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids, position_ids=None, attention_mask=torch.ones_like(input_ids).cuda(),
        past_key_values=None, labels=None, images=images.to(torch.float32)
    )

    print(f"[Debug] Embeddings 形状: {inputs_embeds.shape}")
    print("[Debug] 正在进行前向测试...")
    logits = model(inputs_embeds=inputs_embeds, attention_mask=_attention_mask).logits
    print(f"[Debug] Logits 输出形状: {logits.shape}")

    print("\n>>> 开始正式推理...")
    output_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=_attention_mask,
        do_sample=False,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

# 后处理：去除 Prompt 回显
response = tokenizer.decode(output_ids[0][inputs_embeds.shape[1]:], skip_special_tokens=True).strip()
print("\n" + "=" * 50 + "\n[Qwen3.5 最终模型输出]:\n" + response + "\n" + "=" * 50)