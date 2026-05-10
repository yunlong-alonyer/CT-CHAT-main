import os
import torch
import pandas as pd
import numpy as np
import pydicom
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from typing import Dict, Sequence

# 导入你的模型架构
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX


# =========================================================================
# 1. 鲁棒的 DICOM 预处理函数 )
# =========================================================================
def load_dicom_series(dicom_dir):
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
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
        raise ValueError(f"目录 {dicom_dir} 下未提取到有效图像数据！")

    # ================= 【新增核心防御：剔除定位片与异形切片】 =================
    from collections import Counter
    # 1. 获取所有切片的 2D 维度 (例如 (512, 512))
    shapes = [s.pixel_array.shape for s in slices]
    # 2. 统计出最主流的维度 (也就是真正的 3D 切片序列维度)
    most_common_shape = Counter(shapes).most_common(1)[0][0]
    # 3. 强行过滤：只保留符合主流维度的切片，把异形定位片扔掉
    slices = [s for s in slices if s.pixel_array.shape == most_common_shape]

    if not slices:
        raise ValueError("剔除异形切片后，无有效数据！")
    # ====================================================================

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            pass

    xy_spacing = float(getattr(slices[0], 'PixelSpacing', [1.0, 1.0])[0])

    z_spacing = 0.0
    if len(slices) > 1:
        try:
            z_coords = [float(s.ImagePositionPatient[2]) for s in slices if hasattr(s, 'ImagePositionPatient')]
            unique_z = sorted(list(set(z_coords)))
            if len(unique_z) > 1:
                z_spacing = abs(unique_z[1] - unique_z[0])
        except Exception:
            pass
    if z_spacing <= 0.0:
        z_spacing = float(getattr(slices[0], 'SliceThickness', 1.0))
    if z_spacing <= 0.0:
        z_spacing = 1.0

    slope = float(getattr(slices[0], 'RescaleSlope', 1.0))
    intercept = float(getattr(slices[0], 'RescaleIntercept', -1024.0))

    image_data = np.stack([s.pixel_array for s in slices], axis=-1)
    image_data = np.transpose(image_data, (1, 0, 2))

    return image_data, slope, intercept, xy_spacing, z_spacing


def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


def process_dicom_for_v2(dicom_dir):
    img_data, slope, intercept, xy_spacing, z_spacing = load_dicom_series(dicom_dir)

    target_x_spacing, target_y_spacing, target_z_spacing = 0.75, 0.75, 1.5
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    img_data = np.clip(img_data, -1000, 1000)
    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    img_data = resize_array(tensor, current, target)[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = (img_data / 1000).astype(np.float32)
    tensor = torch.tensor(img_data)

    target_shape = (240, 240, 32)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = max((dh - tensor.size(0)) // 2, 0)
    pad_h_after = max(dh - tensor.size(0) - pad_h_before, 0)
    pad_w_before = max((dw - tensor.size(1)) // 2, 0)
    pad_w_after = max(dw - tensor.size(1) - pad_w_before, 0)
    pad_d_before = max((dd - tensor.size(2)) // 2, 0)
    pad_d_after = max(dd - tensor.size(2) - pad_d_before, 0)

    tensor = F.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
    tensor = tensor.permute(2, 0, 1)

    # 注意这里返回 [Channel=1, Depth=32, Height=240, Width=240]，Batch维度交给DataLoader拼接
    return tensor.unsqueeze(0).half()


# =========================================================================
# 2. 自定义 Dataset 与 DataCollator
# =========================================================================
class CTReportDataset(Dataset):
    def __init__(self, excel_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        # 读取数据并剔除缺少 dicom_path 或 影像所见 的无效数据
        self.df = pd.read_excel(excel_path).dropna(subset=['dicom_path', '影像所见'])

        # 定义固定的提问 Prompt (包含图像占位符)
        self.system_prompt = f"{DEFAULT_IMAGE_TOKEN}\n请详细描述该胸部CT的影像学特征，并给出诊断意见。\n"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dicom_dir = str(row['dicom_path'])
        # 【核心修复】：动态替换无权限的 OSS 路径，指向你有权限的本地路径
        # 请根据你的实际情况核对这两个前缀是否对应！
        old_prefix = "/data/nodesk-aliyun-oss/mnt"
        new_prefix = "/mnt/share_data/CT/ct_dataset_base_260316"

        if dicom_dir.startswith(old_prefix):
            dicom_dir = dicom_dir.replace(old_prefix, new_prefix)

        # 1. 获取目标文本 (拼装影像所见和影像所得)
        findings = str(row['影像所见'])
        conclusion = str(row.get('影像所得', ''))
        target_text = f"影像所见：{findings}\n影像所得：{conclusion}"

        # 2. 加载与预处理 CT 图像
        try:
            image_tensor = process_dicom_for_v2(dicom_dir)
        except Exception as e:
            # 数据集异常兜底：如果某例读取失败，随机返回另一例，防止训练崩溃
            print(f"[警告] 索引 {idx} 数据加载失败，跳过。原因: {e}")
            return self.__getitem__(np.random.randint(len(self)))

        # 3. 构造 input_ids 和 labels
        # 预处理 prompt (这部分不需要计算 loss)
        prompt_ids = tokenizer_image_token(self.system_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        # 预处理目标报告文本，并加上 eos_token 告诉模型说完了
        report_ids = self.tokenizer(
            target_text + self.tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        # 拼接在一起送给模型
        input_ids = torch.cat([prompt_ids, report_ids])

        # 构造 labels：Prompt 部分设为 IGNORE_INDEX (-100)，目标文本保持原样
        # 这样模型的 Loss 只会根据生成报告的准确度来计算
        labels = torch.cat([
            torch.full_like(prompt_ids, IGNORE_INDEX),
            report_ids
        ])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_tensor
        }


class DataCollatorForCTDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        images = [instance["images"] for instance in instances]

        # 动态 Padding，对齐一个 Batch 内长短不一的文本序列
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # 构建 Attention Mask (非 pad 的位置为 1)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 堆叠图像张量
        images = torch.stack(images, dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images
        }


# =========================================================================
# 3. 主训练流程
# =========================================================================
def main():
    # 路径配置
    EXCEL_PATH = "/mnt/share_data/CT/ct_dataset_base_260316/0316_nodesk_ct_chest_1000.xlsx"
    QWEN_DIR = "../../model/Qwen3.5-9B"
    CT_CLIP_PATH = "./checkpoint/CT-CLIP_v2.pt"
    OUTPUT_DIR = "./checkpoints/ct_projector_pretrain_1000"

    print("[*] 正在加载 Tokenizer 与 配置...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print("[*] 正在初始化 LlavaQwenForCausalLM 模型...")
    # 训练时可以开启 bf16 节省显存且稳定梯度
    model = LlavaQwenForCausalLM.from_pretrained(
        QWEN_DIR, config=raw_config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    # 挂载视觉模块
    model.get_model().vision_tower = build_vision_tower(raw_config)
    model.get_model().mm_projector = build_vision_projector(raw_config)

    # 精度桥接：整个大模型 BF16/FP16，视觉塔强制转回 FP32 防止报错
    model.to(torch.bfloat16)
    model.get_model().vision_tower.to(torch.float32)

    # ================= 【核心】：冻结策略 =================
    print("[*] 应用冻结策略：锁定 Vision Tower 和 LLM，只训练 Projector...")
    model.requires_grad_(False)  # 冻结所有
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True  # 解冻适配器

    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] 可训练参数量 (Projector): {trainable_params:,}")
    # ======================================================

    print("[*] 构建 Dataset...")
    dataset = CTReportDataset(EXCEL_PATH, tokenizer)
    data_collator = DataCollatorForCTDataset(tokenizer)

    # 训练参数设置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # 预训练通常 1~3 个 epoch
        per_device_train_batch_size=1,  # 3D 图像极大，建议单卡 Batch=1
        gradient_accumulation_steps=8,  # 累加 8 步模拟 Batch=8 的效果
        learning_rate=1e-3,  # 仅训练随机初始化的网络层时，学习率可以稍大
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,  # 最多保留 3 个 checkpoint
        bf16=True,  # 推荐使用 bf16 进行稳定训练
        dataloader_num_workers=4,  # 多线程预处理图像
        remove_unused_columns=False,  # 【重要】必须设为 False，否则 HF Trainer 会删掉 images 字段
        report_to="none"  # 取消 wandb 报告，保持终端整洁
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n🚀 开始预训练！")
    trainer.train()

    print(f"\n✅ 训练完成！权重已保存至: {OUTPUT_DIR}")
    # 单独保存训练好的 Projector 权重，方便以后加载
    torch.save(model.get_model().mm_projector.state_dict(), f"{OUTPUT_DIR}/projector_weights_final.pth")


if __name__ == "__main__":
    main()