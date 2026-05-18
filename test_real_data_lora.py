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
    target_shape = (224, 224, 32)  # 严格匹配大模型显存安全的 16 层 。
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
CT_CLIP_PATH = "/mnt/huali/ct_dataset_10000/output/CTClip_step_21000_full.pt"
NII_PATH = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data/1000705940001/CT175340_2506621215_5.1_Routine_Chest_0.8_sec.7.5mm.nii.gz"

print(f"[*] 加载配置与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
raw_config = AutoConfig.from_pretrained(QWEN_DIR, trust_remote_code=True)

# 注入多模态参数
multimodal_cfg = {
    "mm_vision_tower": "ctclip",
    "vision_tower_path": CT_CLIP_PATH,
    "mm_projector_type": "coca_pooler",
    "mm_hidden_size": 768,
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

from peft import PeftModel

# 定义微调后的 LoRA 路径
LORA_DIR = "/home/huali/workspace/wzl/CTModel/output/qwen-ct-finetune-lora"
print(f"[*] 准备挂载微调后的模型权重: {LORA_DIR}")

# ---------------------------------------------------------
# 1. 加载微调阶段同步更新的 Projector 权重
# LLaVA 的 LoRA 训练默认将 Projector 保存在 non_lora_trainables.bin 中
# ---------------------------------------------------------
projector_weight_path = os.path.join(LORA_DIR, "non_lora_trainables.bin")
if not os.path.exists(projector_weight_path):
    projector_weight_path = os.path.join(LORA_DIR, "mm_projector.bin")

if os.path.exists(projector_weight_path):
    print(f"[*] 正在加载微调版 Projector 权重: {projector_weight_path}")
    checkpoint = torch.load(projector_weight_path, map_location="cpu")
    state_dict = {}

    # 清洗 Key 名（清洗掉 base_model.model.model.mm_projector. 等前缀）
    for k, v in checkpoint.items():
        if "mm_projector" in k:
            new_key = k.split("mm_projector.")[-1]
            state_dict[new_key] = v

    if len(state_dict) > 0:
        msg = model.get_model().mm_projector.load_state_dict(state_dict, strict=True)
        print(f"[*] 微调版 Projector 权重加载结果: {msg}")
else:
    print(f"[!] 警告：未在 LoRA 目录下找到 Projector 权重，将使用随机初始化的 Projector！")

# ---------------------------------------------------------
# 2. 挂载 Qwen 大脑的 LoRA 权重 (关键！)
# ---------------------------------------------------------
print(f"[*] 正在合并 LoRA 权重到 Qwen 底座...")
model = PeftModel.from_pretrained(
    model,
    LORA_DIR,
    torch_dtype=torch.bfloat16
)
# 【可选优化】：为了让推理速度更快，可以把 LoRA 权重直接融合到底座模型中
# model = model.merge_and_unload()

# ---------------------------------------------------------
# 3. 强制对齐精度与设备
# ---------------------------------------------------------
model.get_model().mm_projector.to(dtype=torch.bfloat16, device="cuda")
model.get_model().vision_tower.to(dtype=torch.float32, device="cuda")

model.cuda()
model.eval()
print("[*] 模型微调权重挂载完毕，可以开始推理！")

# 准备文本输入
#prompt = f" {DEFAULT_IMAGE_TOKEN}\n提示词：前面的是一段经过3d编码器的CT图像。请简单告诉我你看到了什么。"
#prompt = f" {DEFAULT_IMAGE_TOKEN}\n 这是一个患者的CT影像，生成一份详细的医疗报告。"

# 这行代码会去字典里拿到你上面配好的 conv_qwen 模板
conv = conv_templates["qwen"].copy()

# 填入用户的问题（带上图片占位符）
#raw_text = f"{DEFAULT_IMAGE_TOKEN}\n提示词：这是一个患者的CT影像，生成一份医疗报告只包含‘影像所见’和‘影像所得’。"
raw_text = f"{DEFAULT_IMAGE_TOKEN}\n提示词：这张CT影像扫描的是人体的哪个部位？请只回答部位名称（例如：头部、胸部、腹部、脊柱）。"
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
# 执行推理
print("\n>>> 开始生成文本...")
# 确保获取到了停止符
stop_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 151645


from llava.constants import IMAGE_TOKEN_INDEX

# --- 数据流验证 1：Token 层面 ---
print("\n[Debug] --- 数据流验证 ---")
# 查找 LLaVA 内部的硬编码图像占位符 -200
has_image_token = (input_ids == IMAGE_TOKEN_INDEX).any().item()
print(f"[Debug] Prompt 中是否成功编码了 LLaVA Image Token (-200): {has_image_token}")

if not has_image_token:
    print("[Error] 警告：你的 input_ids 中没有图像占位符！")
else:
    print(f"[Debug] 占位符注入成功！包含 -200 的数量: {(input_ids == IMAGE_TOKEN_INDEX).sum().item()}")


with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=False):
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=images,
            do_sample=True,
            temperature=0.2,
            top_p=0.8,
            no_repeat_ngram_size=4,
            pad_token_id=stop_token_id,
            eos_token_id=stop_token_id,   # 🚨 最关键的一行：一旦模型想输出 <|im_end|>，强制让它停下，不许继续联想！
            max_new_tokens=2048,
            use_cache=True
        )



# 解码
input_token_len = input_ids.shape[1]
new_tokens = output_ids[0][input_token_len:].tolist()
response = tokenizer.decode([t for t in new_tokens if t >= 0], skip_special_tokens=True).strip()

# 🌟 新增：自动过滤 think 过程
if "</think>" in response:
    # 按照 </think> 切分，只取后面的正式报告部分
    final_report = response.split("</think>")[-1].strip()
else:
    # 如果模型偶尔没按格式输出，就保留原样
    final_report = response

print("\n" + "=" * 50)
print("[Qwen3.5 最终报告输出]:")
print(final_report)
print("=" * 50)