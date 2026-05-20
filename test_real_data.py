import os
import torch
import numpy as np
import pydicom
import torch.nn.functional as F
import nibabel as nib
import types
from transformers import AutoTokenizer, AutoConfig

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
    """自适应重采样（带 max 保底防御）"""
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [max(1, int(original_shape[i] * scaling_factors[i])) for i in range(len(original_shape))]
    return F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)

def process_nii_for_v2(nii_path):
    """处理 .nii.gz 文件，严格对齐预训练流水线"""
    nii_img = nib.load(str(nii_path))
    img_data = nii_img.get_fdata()

    zooms = nii_img.header.get_zooms()
    xy_spacing = float(zooms[0])
    z_spacing = float(zooms[2])

    while img_data.ndim > 3:
        img_data = img_data[..., 0]
    if img_data.ndim == 2:
        img_data = img_data[:, :, np.newaxis]
    if img_data.ndim < 2:
        raise ValueError(f"严重畸形数据，维度过低: {img_data.shape}")

    slope, intercept = 1.0, 0.0
    img_data = slope * img_data + intercept
    img_data = img_data.transpose(2, 0, 1)

    target_x_spacing, target_y_spacing, target_z_spacing = 0.75, 0.75, 1.5
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    tensor = torch.tensor(img_data.copy()).float().unsqueeze(0).unsqueeze(0)
    tensor = resize_array(tensor, current, target)
    img_data = tensor[0][0].numpy()
    img_data = np.transpose(img_data, (1, 2, 0))

    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = (img_data / 1000.0).astype(np.float32)

    tensor = torch.tensor(img_data)
    target_shape = (480, 480, 40)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1
    )

    tensor = tensor.permute(2, 0, 1)
    # 返回 BFloat16 精度
    return tensor.unsqueeze(0).unsqueeze(0).cuda().bfloat16()


# =========================================================================
# 2. 核心架构配置与挂载
# =========================================================================
QWEN_DIR = "../../model/Qwen3.5-9B"
CT_CLIP_PATH = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500_full.pt"
PROJECTOR_WEIGHTS_PATH = "./checkpoints/test2/checkpoint-26/mm_projector.bin"
NII_PATH = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data/10053105940002/CT163369_1090606624_02_HeadRoutine_Seq.nii.gz"

print(f"[*] 加载配置与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
raw_config = AutoConfig.from_pretrained(QWEN_DIR, trust_remote_code=True)

# 注入多模态参数 (确保 mm_hidden_size 与视觉塔完全对齐)
multimodal_cfg = {
    "mm_vision_tower": "ctclip",
    "vision_tower_path": CT_CLIP_PATH,
    "mm_projector_type": "coca_pooler",
    "mm_hidden_size": 512,         # 🚨 必须是 512
    "hidden_size": 4096,
    "image_token_id": 248056,
}
for k, v in multimodal_cfg.items():
    setattr(raw_config, k, v)

print("[*] 正在初始化 LlavaQwen 模型架构...")
model = LlavaQwenForCausalLM.from_pretrained(
    QWEN_DIR, config=raw_config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)

print("[*] 挂载视觉塔与适配器...")
model.get_model().vision_tower = build_vision_tower(raw_config)
model.get_model().mm_projector = build_vision_projector(raw_config)

# ==========================================
# 3. 权重加载与精度对齐隔离
# ==========================================
if os.path.exists(PROJECTOR_WEIGHTS_PATH):
    print(f"[*] 正在加载预训练适配器权重: {PROJECTOR_WEIGHTS_PATH}")
    checkpoint = torch.load(PROJECTOR_WEIGHTS_PATH, map_location="cpu")
    state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("mm_projector.", "") if k.startswith("mm_projector.") else k
        state_dict[new_key] = v

    msg = model.get_model().mm_projector.load_state_dict(state_dict, strict=True)
    print(f"[*] 适配器权重加载成功！{msg}")
else:
    print(f"[!] 警告：未找到权重文件 {PROJECTOR_WEIGHTS_PATH}")

# 强制分配不同模块的精度隔离
model.get_model().mm_projector.to(dtype=torch.bfloat16, device="cuda")
model.get_model().vision_tower.to(dtype=torch.float32, device="cuda")
model.cuda()
model.eval()

# 🚨 动态补丁：在视觉特征经过 Projector 前执行安全转换
def patched_encode_images(self, images):
    images_f32 = images.to(dtype=torch.float32)
    image_features = self.get_model().get_vision_tower()(images_f32)
    # 强制将特征转换为 bfloat16 迎合 Projector
    image_features = image_features.to(dtype=torch.bfloat16)
    image_features = self.get_model().mm_projector(image_features)
    return image_features

model.encode_images = types.MethodType(patched_encode_images, model)

# =========================================================================
# 4. 对话模板、Token 准备与影像处理
# =========================================================================
conv = conv_templates["qwen"].copy()
conv.system = "你是一个专业的医疗AI助手CT-CHAT。请直接按照用户的指令作答，不要输出任何思考过程或解释语。"

# -------------------------------------------------------------
# 模式 A: 基础视力测试（推荐刚练完新 Projector 时测试使用）
# -------------------------------------------------------------
raw_text = f"{DEFAULT_IMAGE_TOKEN}\n这是一张CT影像，请简要告诉我你在影像中看到了什么器官或组织？"

# -------------------------------------------------------------
# 模式 B: 严格的单选题模式（待视力测试通过后切换此模式）
# -------------------------------------------------------------
# raw_text = f"{DEFAULT_IMAGE_TOKEN}\n根据影像特征，该扫描部位属于：A.胸部 B.头部 C.腹部。\n【指令】：直接且仅输出一个代表选项的大写字母，严禁任何问候、解释、分析或思考过程。"
# -------------------------------------------------------------

conv.append_message(conv.roles[0], raw_text)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
attention_mask = torch.ones_like(input_ids).cuda()

print(f"[*] 正在预处理 NIfTI 影像...")
images = process_nii_for_v2(NII_PATH)
print(f"[*] 预处理完成，张量形状: {images.shape}，精度: {images.dtype}")

# 强制压制思考标签
bad_words = ["<think>", "</think>", "Thinking Process:", "思考过程"]
bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in bad_words]
bad_words_ids = [ids for ids in bad_words_ids if len(ids) > 0]
stop_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 151645

# =========================================================================
# 5. 组装变长特征与自回归生成 (增强版)
# =========================================================================
print("\n[Debug] 正在拼装特征...")
images_f32 = images.to(device="cuda", dtype=torch.float32)

with torch.no_grad():
    # 显式使用 prepare_inputs_labels_for_multimodal
    (
        _input_ids, _position_ids, _attention_mask,
        _past_key_values, inputs_embeds, _labels
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=images_f32
    )

    print(f"[Debug] Embeddings 形状: {inputs_embeds.shape}, 设备: {inputs_embeds.device}")

    # 强制进行一次简易的前向推理，检查是否会报错
    print("[Debug] 正在进行前向测试...")
    logits = model(inputs_embeds=inputs_embeds, attention_mask=_attention_mask).logits
    print(f"[Debug] Logits 输出形状: {logits.shape} (模型已处理完所有 Token)")

    # 开始生成
    print("\n>>> 开始正式推理...")
    output_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=_attention_mask,
        do_sample=False,  # 暂时关闭采样，使用贪心策略
        max_new_tokens=512,  # 只要能吐出 20 个字，证明通了
        pad_token_id=stop_token_id,
        eos_token_id=stop_token_id,
        use_cache=True
    )

# =========================================================================
# 6. 解码与输出
# =========================================================================
# 直接从生成的全部 sequences 中取出生成的回答部分
# 因为我们使用了 inputs_embeds 喂入，部分版本的 generate 会返回完整序列
generated_tokens = output_ids[0]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print("\n" + "=" * 50)
print("[Qwen3.5 最终模型输出]:")
print(response)
print("=" * 50)