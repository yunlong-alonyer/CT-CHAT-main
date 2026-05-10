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


# =========================================================================
# 1. DICOM 序列读取与预处理函数
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
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    return F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)


def process_dicom_for_v2(dicom_dir):
    """
    处理 DICOM 并对其进行重采样、裁剪、填充
    """
    img_data, slope, intercept, xy_spacing, z_spacing = load_dicom_series(dicom_dir)

    # 1. 重采样
    target_x_spacing, target_y_spacing, target_z_spacing = 0.75, 0.75, 1.5
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    img_data = np.clip(img_data, -1000, 1000)
    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data).float().unsqueeze(0).unsqueeze(0)
    tensor = resize_array(tensor, current, target)
    img_data = tensor.squeeze().numpy()
    img_data = np.transpose(img_data, (1, 2, 0))

    # 2. 归一化
    img_data = (img_data / 1000).astype(np.float32)
    tensor = torch.tensor(img_data)

    # 3. 【核心对齐】: 必须是 224 以匹配视觉塔
    target_shape = (224, 224, 32)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    h_start, w_start, d_start = max((h - dh) // 2, 0), max((w - dw) // 2, 0), max((d - dd) // 2, 0)
    tensor = tensor[h_start:h_start + dh, w_start:w_start + dw, d_start:d_start + dd]

    pad_h = max(dh - tensor.size(0), 0)
    pad_w = max(dw - tensor.size(1), 0)
    pad_d = max(dd - tensor.size(2), 0)
    tensor = F.pad(tensor, (0, pad_d, 0, pad_w, 0, pad_h), value=-1)

    # 4. 最终形状 [1, 1, 32, 224, 224]
    tensor = tensor.permute(2, 0, 1)
    return tensor.unsqueeze(0).unsqueeze(0).cuda().bfloat16()


# =========================================================================
# 2. 模型挂载与推理
# =========================================================================

QWEN_DIR = "../../model/Qwen3.5-9B"
CT_CLIP_PATH = "checkpoint/CT-CLIP_v2.pt"
DICOM_DIR = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000/10050302800015"

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

PROJECTOR_WEIGHTS_PATH = "./checkpoints/qwen-ct-pretrain/mm_projector.bin"
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
    # 建议先设置 strict=False 看看能否跑通，如果没问题再改回 True
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

# 准备文本输入
prompt = f" {DEFAULT_IMAGE_TOKEN}\n提示词：前面的是一段经过3d编码器的CT图像。请简单告诉我你看到了什么。"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

# 准备影像
print(f"[*] 正在预处理 DICOM 序列...")
images = process_dicom_for_v2(DICOM_DIR)
print(f"[*] 预处理完成，张量形状: {images.shape}，精度: {images.dtype}")

# 执行推理
print("\n>>> 开始生成文本...")
with torch.no_grad():
    # 显式关闭 autocast 以配合视觉塔的隔离岛策略
    with torch.amp.autocast('cuda', enabled=False):
        output_ids = model.generate(
            input_ids, images=images, do_sample=True, temperature=0.7, max_new_tokens=256, use_cache=True
        )

# 解码
input_token_len = input_ids.shape[1]
new_tokens = output_ids[0][input_token_len:].tolist()
response = tokenizer.decode([t for t in new_tokens if t >= 0], skip_special_tokens=True).strip()

print("\n" + "=" * 50)
print("[Qwen3.5 推理输出]:")
print(response)
print("=" * 50)