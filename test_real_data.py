import os
import torch
import numpy as np
import pydicom
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

# 引入你的模型架构
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import nibabel as nib

# =========================================================================
# 1. DICOM 序列读取与预处理函数12
# =========================================================================
def load_dicom_series(dicom_dir):
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    if not files:
        raise ValueError(f"在目录 {dicom_dir} 下未找到任何文件！")

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"目录 {dicom_dir} 下没有有效 DICOM 数据！")

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    xy_spacing = float(slices[0].PixelSpacing[0]) if hasattr(slices[0], 'PixelSpacing') else 1.0

    z_coords = sorted(
        list(set([float(s.ImagePositionPatient[2]) for s in slices if hasattr(s, 'ImagePositionPatient')])))
    z_spacing = abs(z_coords[1] - z_coords[0]) if len(z_coords) > 1 else float(
        getattr(slices[0], 'SliceThickness', 1.0))
    if z_spacing <= 0: z_spacing = 1.0

    slope = float(getattr(slices[0], 'RescaleSlope', 1.0))
    intercept = float(getattr(slices[0], 'RescaleIntercept', -1024.0))

    image_data = np.stack([s.pixel_array for s in slices], axis=-1)
    image_data = np.transpose(image_data, (1, 0, 2))
    return image_data, slope, intercept, xy_spacing, z_spacing





def resize_array(array, current_spacing, target_spacing):
    """自适应重采样（带 max 保底防御）"""
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [max(1, int(original_shape[i] * scaling_factors[i])) for i in range(len(original_shape))]
    return F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)


def process_nii_for_v2(nii_path):
    """
    处理 .nii.gz 文件，严格对齐预训练流水线
    """
    # 1. 读取 NIfTI 文件及头信息
    nii_img = nib.load(str(nii_path))
    img_data = nii_img.get_fdata()

    # 获取体素间距 (通常 zooms 返回 [x, y, z])
    zooms = nii_img.header.get_zooms()
    xy_spacing = float(zooms[0])
    z_spacing = float(zooms[2])

    # 🚨 防御机制：强制降维到真正的 3D
    while img_data.ndim > 3:
        img_data = img_data[..., 0]
    if img_data.ndim == 2:
        img_data = img_data[:, :, np.newaxis]
    if img_data.ndim < 2:
        raise ValueError(f"严重畸形数据，维度过低: {img_data.shape}")

    # 默认你的 .nii.gz 已经保存为了 HU 值，或者不需特殊换算
    # 这与 train.py 中查不到 CSV 时的 fallback 逻辑完全一致
    slope, intercept = 1.0, 0.0
    img_data = slope * img_data + intercept

    # 转为 [Depth, Height, Width] 准备重采样 (完全对齐 train.py)
    img_data = img_data.transpose(2, 0, 1)

    # 2. 空间重采样
    target_x_spacing, target_y_spacing, target_z_spacing = 0.75, 0.75, 1.5
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    tensor = torch.tensor(img_data.copy()).float().unsqueeze(0).unsqueeze(0)
    tensor = resize_array(tensor, current, target)
    img_data = tensor[0][0].numpy()

    # 转回 [Height, Width, Depth]
    img_data = np.transpose(img_data, (1, 2, 0))

    # 3. 截断与归一化到 [-1, 1]
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = (img_data / 1000.0).astype(np.float32)

    tensor = torch.tensor(img_data)

    # 4. 尺寸对齐与居中裁剪
    target_shape = (480, 480, 40)  # 严格匹配大模型显存安全的 16 层 。
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # 5. 居中填充 (填充值为无意义空气 -1)
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

    # 6. 转为 [1, 1, Depth, Height, Width] -> [1, 1, 16, 224, 224]
    tensor = tensor.permute(2, 0, 1)
    return tensor.unsqueeze(0).unsqueeze(0).cuda().bfloat16()


# =========================================================================
# 2. 模型挂载与推理
# =========================================================================

QWEN_DIR = "../../model/Qwen3.5-9B"
CT_CLIP_PATH = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500_full.pt"
#CT_CLIP_PATH = "./checkpoint/CT-CLIP_v2.pt"
NII_PATH = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data/10053105940002/CT163369_1090606624_02_HeadRoutine_Seq.nii.gz"

print(f"[*] 加载配置与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
raw_config = AutoConfig.from_pretrained(QWEN_DIR, trust_remote_code=True)

# 注入多模态参数
multimodal_cfg = {
    "mm_vision_tower": "ctclip",
    "vision_tower_path": CT_CLIP_PATH,
    "mm_projector_type": "coca_pooler",
    "mm_hidden_size": 512,
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

PROJECTOR_WEIGHTS_PATH = "/mnt/huali/checkpoint_projector_34500_2/checkpoint-15514/mm_projector.bin"
#PROJECTOR_WEIGHTS_PATH =  ""
# ==========================================
# 修改 test_real_data.py 中加载适配器权重的部分
# ==========================================

if os.path.exists(PROJECTOR_WEIGHTS_PATH):
    print(f"[*] 正在加载预训练适配器权重: {PROJECTOR_WEIGHTS_PATH}")

    # 1. 加载原始权重
    checkpoint = torch.load(PROJECTOR_WEIGHTS_PATH, map_location="cpu")

    # 2. 精准清洗 Key 名
    # 根据报错信息，前缀是 "mm_projector."
    state_dict = {}
    for k, v in checkpoint.items():
        # 如果 Key 开头是 mm_projector.，则去掉它
        if k.startswith("mm_projector."):
            new_key = k.replace("mm_projector.", "")
            state_dict[new_key] = v
        else:
            state_dict[k] = v

    # 3. 加载到模型中
    msg = model.get_model().mm_projector.load_state_dict(state_dict, strict=True)
    print(f"[*] 适配器权重加载结果: {msg}")
    print("[*] 适配器权重加载成功！")
else:
    print(f"[!] 警告：未找到权重文件 {PROJECTOR_WEIGHTS_PATH}")

# --- 核心修复：强制对齐精度与设备 ---
# 1. 适配器（Projector）必须跟随 LLM 使用 bfloat16
model.get_model().mm_projector.to(dtype=torch.bfloat16, device="cuda")

# 2. 视觉塔使用隔离岛策略，强制保持在 float32
model.get_model().vision_tower.to(dtype=torch.float32, device="cuda")

model.cuda()
model.eval()

# =========================================================================
# [核心修复] 动态补丁：在视觉塔和适配器之间架设精度转换桥梁
# =========================================================================
import types


def patched_encode_images(self, images):
    # 1. 经过 Vision Tower (此时图像特征还是 float32)
    image_features = self.get_model().get_vision_tower()(images)

    # 2. 🚨 核心转换：强制将特征转换为 bfloat16，迎合 mm_projector 和 LLM
    image_features = image_features.to(dtype=torch.bfloat16)

    # 3. 经过 Projector (安全通过)
    image_features = self.get_model().mm_projector(image_features)

    return image_features


# 将打好补丁的方法强制绑定给当前模型
model.encode_images = types.MethodType(patched_encode_images, model)
# =========================================================================

# 准备文本输入
#prompt = f" {DEFAULT_IMAGE_TOKEN}\n提示词：前面的是一段经过3d编码器的CT图像。请简单告诉我你看到了什么。"
#prompt = f" {DEFAULT_IMAGE_TOKEN}\n 这是一个患者的CT影像，生成一份详细的医疗报告。"

# 这行代码会去字典里拿到你上面配好的 conv_qwen 模板
conv = conv_templates["qwen"].copy()
conv.system = "你是一个专业的医疗AI助手CT-CHAT。请直接根据提供的CT影像作答"
# 填入用户的问题（带上图片占位符）
#raw_text = f"{DEFAULT_IMAGE_TOKEN}\n提示词：这是一个患者的CT影像，生成一份医疗报告只包含‘影像所见’和‘影像所得’。"
#raw_text = f"{DEFAULT_IMAGE_TOKEN}\n根据前文的医学特征提示，作为专业的放射科医生，请解读这份CT影像，提供影像所见及结论"
raw_text = f"{DEFAULT_IMAGE_TOKEN}\n根据影像特征，该扫描部位属于：A.胸部 B.头部 C.腹部。\n【指令】：直接且仅输出一个代表选项的大写字母，严禁任何问候、解释、分析或思考过程。"
conv.append_message(conv.roles[0], raw_text)
conv.append_message(conv.roles[1], None)

# 生成最终的 Prompt 字符串
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

# 准备影像
print(f"[*] 正在预处理 NIfTI 影像...")
images = process_nii_for_v2(NII_PATH)
print(f"[*] 预处理完成，张量形状: {images.shape}，精度: {images.dtype}")
attention_mask = torch.ones_like(input_ids).cuda()

# 强制模型在生成时遇到这些词的概率降为 0
bad_words = ["<think>", "</think>", "Thinking Process:", "思考过程"]
bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in bad_words]

# 过滤掉可能为空的 encode 结果
bad_words_ids = [ids for ids in bad_words_ids if len(ids) > 0]

# -------------------------------------------------------------------
# [新增] 核心探针：验证图像数据是否真正在模型内部流通
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# [Debug & 修复] 提前手动进行多模态特征融合
# -------------------------------------------------------------------
# =========================================================================
# [核心修复] 动态补丁：在视觉塔和适配器之间架设精度转换桥梁
# =========================================================================
# =========================================================================
# [核心修复 1] 动态补丁：架设精度转换桥梁
# =========================================================================
import types
def patched_encode_images(self, images):
    images_f32 = images.to(dtype=torch.float32)
    image_features = self.get_model().get_vision_tower()(images_f32)
    image_features = image_features.to(dtype=torch.bfloat16)
    image_features = self.get_model().mm_projector(image_features)
    return image_features

model.encode_images = types.MethodType(patched_encode_images, model)

# =========================================================================
# [核心修复 2] 屏蔽词与对话模板设置
# =========================================================================
conv = conv_templates["qwen"].copy()
conv.system = "你是一个专业的医疗AI助手CT-CHAT。请直接描述影像内容，不要输出任何思考过程或解释语。"
raw_text = f"{DEFAULT_IMAGE_TOKEN}\n这是一张CT影像，请简要告诉我你在影像中看到了什么？"
conv.append_message(conv.roles[0], raw_text)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
attention_mask = torch.ones_like(input_ids).cuda()

bad_words = ["<think>", "</think>", "Thinking Process:", "思考过程"]
bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in bad_words]
bad_words_ids = [ids for ids in bad_words_ids if len(ids) > 0]
stop_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 151645

# =========================================================================
# [核心修复 3] 强行手动拉长多模态 Token 与 Attention Mask
# =========================================================================
print("\n[Debug] 正在拼装特征并修正 Attention Mask...")
images_f32 = images.to(device="cuda", dtype=torch.float32)
(
    _input_ids,
    _position_ids,
    _attention_mask,     # 🚨 拿到了长度为 328 的正确掩码！
    _past_key_values,
    inputs_embeds,       # 🚨 拿到了 328 个真实的向量！
    _labels
) = model.prepare_inputs_labels_for_multimodal(
    input_ids=input_ids,
    position_ids=None,
    attention_mask=attention_mask,
    past_key_values=None,
    labels=None,
    images=images_f32
)

# =========================================================================
# [核心修复 4] 基于修正后的特征执行推理
# =========================================================================
print("\n>>> 开始基于修复后特征的推理...")
with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=False):
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,       # 直接喂特征
            attention_mask=_attention_mask,    # 直接喂修复好的变长 Mask
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=stop_token_id,
            eos_token_id=stop_token_id,
            bad_words_ids=bad_words_ids,       # 强制压制 <think>
            max_new_tokens=256,
            use_cache=True
        )

# =========================================================================
# [核心修复 5] 解码与输出
# =========================================================================
# 由于喂入的是 inputs_embeds，生成的只有答案，直接解码即可
new_tokens = output_ids[0].tolist()
response = tokenizer.decode([t for t in new_tokens if t >= 0], skip_special_tokens=True).strip()

if response.startswith("assistant"):
    response = response[len("assistant"):].strip()

print("\n" + "=" * 50)
print("[Qwen3.5 最终报告输出]:")
print(response)
print("=" * 50)