import os
import torch
import numpy as np
import pydicom
import torch.nn.functional as F
import nibabel as nib
import types
import re
from transformers import AutoTokenizer, AutoConfig

# 引入模型架构
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


# =========================================================================
# 1. 影像预处理函数
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
    target_x, target_y, target_z = 0.75, 0.75, 1.5
    tensor = torch.tensor(img_data.copy()).float().unsqueeze(0).unsqueeze(0)
    tensor = resize_array(tensor, (z_spacing, xy_spacing, xy_spacing), (target_z, target_x, target_y))
    img_data = np.transpose(tensor[0][0].numpy(), (1, 2, 0))

    img_data = np.clip(img_data, -1000, 1000) / 1000.0
    tensor = torch.tensor(img_data.astype(np.float32))

    # 尺寸对齐 480x480x40
    target_shape = (480, 480, 40)
    h, w, d = tensor.shape
    h_s, w_s, d_s = max((h - target_shape[0]) // 2, 0), max((w - target_shape[1]) // 2, 0), max(
        (d - target_shape[2]) // 2, 0)
    tensor = tensor[h_s:h_s + target_shape[0], w_s:w_s + target_shape[1], d_s:d_s + target_shape[2]]

    pad = ((target_shape[2] - tensor.size(2)) // 2, (target_shape[2] - tensor.size(2) + 1) // 2,
           (target_shape[1] - tensor.size(1)) // 2, (target_shape[1] - tensor.size(1) + 1) // 2,
           (target_shape[0] - tensor.size(0)) // 2, (target_shape[0] - tensor.size(0) + 1) // 2)
    tensor = torch.nn.functional.pad(tensor, pad, value=-1)
    return tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).cuda().bfloat16()


# ======================================================================1===
# 2. 模型初始化
# =========================================================================
QWEN_DIR = "../../model/Qwen3.5-9B"
CT_CLIP_PATH = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500_full.pt"
#CT_CLIP_PATH = "/mnt/huali/checkpoint/CT-CLIP_v2.pt"
PROJECTOR_WEIGHTS_PATH = "/mnt/huali/checkpoint_projector_34500_4/checkpoint-485/mm_projector.bin"
NII_PATH = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data/100002300082/000972_1305455278_2_L_SpineRoutine.nii.gz"

print(f"\n[Info] 正在初始化 Tokenizer 和模型配置...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
raw_config = AutoConfig.from_pretrained(QWEN_DIR, trust_remote_code=True)
multimodal_cfg = {"mm_vision_tower": "ctclip", "vision_tower_path": CT_CLIP_PATH, "mm_projector_type": "coca_pooler",
                  "mm_hidden_size": 512, "hidden_size": 4096, "image_token_id": 248056}
for k, v in multimodal_cfg.items(): setattr(raw_config, k, v)

print(f"[Info] 正在加载 Qwen 基座模型与 CT-CLIP 视觉编码器...")
model = LlavaQwenForCausalLM.from_pretrained(QWEN_DIR, config=raw_config, torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
model.get_model().vision_tower = build_vision_tower(raw_config)
model.get_model().mm_projector = build_vision_projector(raw_config)

# 加载权重
print(f"[Info] 正在从 {PROJECTOR_WEIGHTS_PATH} 加载 Projector 适配器权重...")
checkpoint = torch.load(PROJECTOR_WEIGHTS_PATH, map_location="cpu")
state_dict = {k.replace("mm_projector.", ""): v for k, v in checkpoint.items()}
model.get_model().mm_projector.load_state_dict(state_dict, strict=True)
print(f"[Info] Projector 适配器权重加载成功！")

# 挂载设备
model.get_model().mm_projector.to(dtype=torch.bfloat16, device="cuda")
model.get_model().vision_tower.to(dtype=torch.float32, device="cuda")
model.cuda().eval()

# ===== 新增：打印模型加载成功后的总览及维度信息 START =====
print("\n" + "=" * 60)
print("🚀 [组件加载成功概览与特征维度说明]")
print(f"1. 语言模型 (LLM): Qwen3.5-9B")
print(f"   - 隐藏层维度 (Hidden Size): {model.config.hidden_size}")
print(f"   - 运行精度: {model.dtype}")
print(f"2. 视觉编码器 (Vision Tower): CT-CLIP")
print(f"   - 权重路径: {CT_CLIP_PATH}")
print(f"   - 预期输入维度: [Batch, Channel(1), Depth(40), Height(480), Width(480)]")
print(f"   - 预期输出维度: [Batch, Patch_num(2304), Feature_dim({model.config.mm_hidden_size})]")
print(f"3. 多模态适配器 (Projector): {model.config.mm_projector_type}")
print(f"   - 预期输入维度: [Batch, 2304, {model.config.mm_hidden_size}]")
print(f"   - 预期输出维度: [Batch, Output_Tokens(256), LLM_Hidden({model.config.hidden_size})]")
print("=" * 60 + "\n")
# ===== 新增：打印模型加载成功后的总览及维度信息 END =====

# 精度桥梁补丁
def patched_encode_images(self, images):
    images_f32 = images.to(dtype=torch.float32)
    image_features = self.get_model().get_vision_tower()(images_f32)
    return self.get_model().mm_projector(image_features.to(dtype=torch.bfloat16))


model.encode_images = types.MethodType(patched_encode_images, model)

# =========================================================================
# 3. 推理逻辑
# =========================================================================
conv = conv_templates["qwen"].copy()
conv.system = "你是一个专业的医学影像诊断专家，可以通过传入的图像特征直接看到CT影像。"
raw_text = f"{DEFAULT_IMAGE_TOKEN}\n作为专业的放射科医生，请解读这份3D CT影像，提供影像所见及结论。"
conv.append_message(conv.roles[0], raw_text)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()
print(f"\n[Debug] 最终送入模型的文本 Prompt:\n{prompt}")

input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
    0).cuda()

print("\n[Debug] 正在ccccccccccccc...")
images = process_nii_for_v2(NII_PATH)

img2 = torch.zeros_like(images)
img3 = torch.randn_like(images)
with torch.no_grad():
    out1 = model.generate(input_ids, images=images, do_sample=False, max_new_tokens=128)
    out2 = model.generate(input_ids, images=img2, do_sample=False, max_new_tokens=128)
    out3 = model.generate(input_ids, images=img3, do_sample=False, max_new_tokens=128)

print("真实图：\n"+tokenizer.decode(out1[0][input_ids.shape[1]:],skip_special_tokens=True)+"\n")
print("全黑图：\n"+tokenizer.decode(out2[0][input_ids.shape[1]:],skip_special_tokens=True)+"\n")
print("随机图：\n"+tokenizer.decode(out3[0][input_ids.shape[1]:],skip_special_tokens=True))


print("\n[Debug] 正在拼装特征...")

# ===== 新增诊断代码 START =====
with torch.no_grad():
    # 第一步：验证 CT-CLIP 视觉特征
    images_f32 = images.to(dtype=torch.float32)
    vision_features = model.get_model().get_vision_tower()(images_f32)
    print(f"Vision features shape: {vision_features.shape}")
    print(f"Vision features mean: {vision_features.mean():.4f}")
    print(f"Vision features std: {vision_features.std():.4f}")
    print(f"Vision features min/max: {vision_features.min():.4f} / {vision_features.max():.4f}")

    # 第二步：验证 projector 输出
    proj_features = model.get_model().mm_projector(vision_features.to(dtype=torch.bfloat16))
    print(f"Projector output shape: {proj_features.shape}")
    print(f"Projector output std: {proj_features.std():.4f}")
    print(f"Projector output mean: {proj_features.mean():.4f}")

# 第三步：纯文字推理对比
text_only = "这是一张CT影像，请简要告诉我你在影像中看到了什么器官或组织？"
text_ids = tokenizer(text_only, return_tensors='pt').input_ids.cuda()
with torch.no_grad():
    # 最保险：让模型自己处理纯文字
    output_text_only = model.generate(
        input_ids=text_ids,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
    )
print("纯文字输出:", tokenizer.decode(output_text_only[0], skip_special_tokens=True))
# ===== 新增诊断代码 END =====


with torch.no_grad():
    (
        _input_ids, _position_ids, _attention_mask,
        _past_key_values, inputs_embeds, _labels
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids, position_ids=None, attention_mask=torch.ones_like(input_ids),
        past_key_values=None, labels=None, images=images.to(torch.float32)
    )

    print(f"[Debug] Embeddings 形状: {inputs_embeds.shape}, 设备: {inputs_embeds.device}")
    print("[Debug] 正在进行前向测试...")
    logits = model(inputs_embeds=inputs_embeds, attention_mask=_attention_mask).logits
    print(f"[Debug] Logits 输出形状: {logits.shape} (模型已处理完所有 Token)")

    print("\n>>> 开始正式推理...")
    output_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=_attention_mask,
        do_sample=True, temperature=0.2, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        use_cache=True
    )

# 修复解码部分
response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# 过滤 think 块（保留思考，但不输出）
response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

# 同时去掉可能残留的 im_end
response = response.replace('<|im_end|>', '').strip()

print("\n" + "=" * 50)
print("[Qwen3.5 最终模型输出]:")
print(response)
print("=" * 50)