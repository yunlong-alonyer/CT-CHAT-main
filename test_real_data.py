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
    """
    读取文件夹下的所有 DICOM 文件，并自动提取 3D 体素矩阵和元数据
    增加了强行读取(force=True)和非法文件过滤逻辑。
    """
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    if not files:
        raise ValueError(f"在目录 {dicom_dir} 下未找到任何文件！")

    slices = []
    for f in files:
        try:
            # 核心修改：使用 force=True 强行跳过缺失的文件头
            ds = pydicom.dcmread(f, force=True)
            # 强制读取可能导致元数据确实，手动设置小端传输语法
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

            # 只保留真正包含图像像素数据的文件
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            # 忽略无法解析的隐藏文件或系统日志文件
            continue

    if not slices:
        raise ValueError(f"目录 {dicom_dir} 下没有提取到任何包含图像数据的有效 DICOM 切片！")

    # 2. 根据 Z 轴坐标 (ImagePositionPatient[2]) 或 InstanceNumber 对切片进行物理排序
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            print("[警告] 无法找到 Z 轴坐标或层序号，将按照文件读取顺序排列，这可能导致 3D 图像断层！")

    # 3. 提取 Spacing (XY 轴体素间距)
    try:
        xy_spacing = float(slices[0].PixelSpacing[0])
    except AttributeError:
        xy_spacing = 1.0  # 默认兜底

    # 4. 提取 Z 轴间距 (层厚) - 终极防御版
    z_spacing = 0.0
    if len(slices) > 1:
        try:
            # 改进：获取所有切片的 Z 坐标，去重并排序，计算真实的层间距
            z_coords = [float(s.ImagePositionPatient[2]) for s in slices if hasattr(s, 'ImagePositionPatient')]
            unique_z = sorted(list(set(z_coords)))
            if len(unique_z) > 1:
                # 取相邻两个不同 Z 坐标的差值
                z_spacing = abs(unique_z[1] - unique_z[0])
        except Exception:
            pass

    # 兜底机制 1：如果物理坐标算不出来，或者算出来是 0，读取字典里的 SliceThickness
    if z_spacing == 0.0:
        z_spacing = float(getattr(slices[0], 'SliceThickness', 1.0))

    # 终极兜底 2：如果连 SliceThickness 也是 0 或者异常负数，强制设为常规 CT 层厚 1.0
    if z_spacing <= 0.0:
        print("[警告] 无法解析有效的 Z 轴层厚，强行默认层厚为 1.0 mm。")
        z_spacing = 1.0

    # 5. 提取转换真实 HU 值所需的斜率和截距
    slope = float(getattr(slices[0], 'RescaleSlope', 1.0))
    intercept = float(getattr(slices[0], 'RescaleIntercept', -1024.0))

    # 6. 堆叠为 3D Numpy 数组 [Height, Width, Depth]
    image_data = np.stack([s.pixel_array for s in slices], axis=-1)

    # 转换为与 NIfTI 一致的 (X, Y, Z) 格式
    image_data = np.transpose(image_data, (1, 0, 2))

    return image_data, slope, intercept, xy_spacing, z_spacing

def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


def process_dicom_for_v2(dicom_dir):
    """
    处理 DICOM 并对其进行重采样、裁剪、填充，以迎合 CT-CLIP_v2
    """
    # 1. 动态加载 DICOM 数据
    img_data, slope, intercept, xy_spacing, z_spacing = load_dicom_series(dicom_dir)

    # 2. 重采样目标 Spacing
    target_x_spacing, target_y_spacing, target_z_spacing = 0.75, 0.75, 1.5
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    # 3. 窗宽窗位截断 (CT HU 值)
    img_data = slope * img_data + intercept
    img_data = np.clip(img_data, -1000, 1000)
    img_data = img_data.transpose(2, 0, 1)  # 转为 (Z, X, Y) 进行重采样

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    img_data = resize_array(tensor, current, target)[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))  # 转回 (X, Y, Z)

    # 4. 归一化 [-1000, 1000] -> [-1, 1]
    img_data = (img_data / 1000).astype(np.float32)
    tensor = torch.tensor(img_data)

    # 5. 【核心对齐】: CT-CLIP_v2 目标输入尺寸是 (240, 240, 32) [X, Y, Z]
    target_shape = (224, 224, 32)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    # 中心裁剪
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # 边缘填充 (值为 -1，代表空气)
    pad_h_before = max((dh - tensor.size(0)) // 2, 0)
    pad_h_after = max(dh - tensor.size(0) - pad_h_before, 0)
    pad_w_before = max((dw - tensor.size(1)) // 2, 0)
    pad_w_after = max(dw - tensor.size(1) - pad_w_before, 0)
    pad_d_before = max((dd - tensor.size(2)) // 2, 0)
    pad_d_after = max(dd - tensor.size(2) - pad_d_before, 0)

    tensor = F.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    # 6. 转换维度为 [Depth, Height, Width] -> [32, 240, 240]
    tensor = tensor.permute(2, 0, 1)

    # 7. 最终增加 Batch 和 Channel 维度: [1, 1, 32, 240, 240]，并转半精度
    #return tensor.unsqueeze(0).unsqueeze(0).cuda().half()
    # 7. 最终增加 Batch 和 Channel 维度，并转为与模型一致的 BFloat16
    return tensor.unsqueeze(0).unsqueeze(0).cuda().bfloat16()


# =========================================================================
# 2. 模型挂载与推理
# =========================================================================

QWEN_DIR = "../../model/Qwen3.5-9B"
CT_CLIP_PATH = "checkpoint/CT-CLIP_v2.pt"

# 【重要配置】指向你存放某个患者 DICOM 序列的文件夹路径
DICOM_DIR = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000/10050302800015"

print(f"[*] 加载配置与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
raw_config = AutoConfig.from_pretrained(QWEN_DIR, trust_remote_code=True)

# 注入多模态参数 (使用 CT-CLIP_v2 配置)
multimodal_cfg = {
    "mm_vision_tower": "ctclip",
    "vision_tower_path": CT_CLIP_PATH,
    "mm_projector_type": "coca_pooler",
    "mm_hidden_size": 768,  # v2 为 768
    "hidden_size": 4096,
    "image_token_id": 248056,
}
for k, v in multimodal_cfg.items():
    setattr(raw_config, k, v)

print("[*] 正在初始化 123LlavaQwenForCausalLM 模型架构...")
model = LlavaQwenForCausalLM.from_pretrained(
    QWEN_DIR, config=raw_config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)

print("[*] 挂载视觉塔与适配器...")
model.get_model().vision_tower = build_vision_tower(raw_config)
model.get_model().mm_projector = build_vision_projector(raw_config)

# 精度桥接策略：模型整体 FP16，视觉塔单兵 FP32
#model.to(dtype=torch.float16, device="cuda")
#model.get_model().vision_tower.to(torch.float32)

# 改为直接推入 CUDA，保持统一的 bfloat16 精度：
model.cuda()
model.eval()

# 准备文本输入
#prompt = f"{DEFAULT_IMAGE_TOKEN}\nPlease provide a detailed diagnostic report for this 3D CT scan."
prompt = f" {DEFAULT_IMAGE_TOKEN}\n，提示词：前面的是一段经过3d编码器的CT图像你看到的可能是一段乱码 先简单告诉我你看到了什么."
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

# 准备真实的 DICOM 图像输入
print(f"[*] 正在预处理真实的 DICOM 序列数据: {DICOM_DIR} ...")
try:
    images = process_dicom_for_v2(DICOM_DIR)
    print(f"[*] 真实影像预处理完成，张量形状: {images.shape} (期望: [1, 1, 32, 240, 240])")
except Exception as e:
    print(f"[!] 读取或预处理影像失败: {e}")
    exit(1)

# 执行推理
print("\n>>> 开始端到端前向传播并生成文本...")
with torch.no_grad():
    output_ids = model.generate(
        input_ids, images=images, do_sample=True, temperature=0.7, max_new_tokens=256, use_cache=True
    )

# 截取并解码生成的文本
input_token_len = input_ids.shape[1]
new_tokens = output_ids[0][input_token_len:].tolist()
valid_tokens = [t for t in new_tokens if t >= 0]
response = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()

print("\n" + "=" * 50)
print("[Qwen3.5 DICOM 影像推理输出]:")
print(response)
print("=" * 50)