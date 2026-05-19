import torch
import torch.nn as nn
import os
# 注意：确保你的环境里能正确 import transformer_maskgit
from transformer_maskgit.ctvit import CTViT


class CTCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_name, args, **kwargs):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        self.vision_tower_path = getattr(args, 'vision_tower_path', None)

        # 兜底逻辑：如果配置里没写 vision_tower_path，再把名字当路径试一试
        if self.vision_tower_path is None:
            self.vision_tower_path = vision_tower_name

        # 2. 初始化 CT-CLIP v2 核心架构
        # 注意：image_size 必须固定为 224 以匹配权重
        self.vision_tower = CTViT(
            dim=768,
            codebook_size=8192,
            image_size=224,
            patch_size=16,
            temporal_patch_size=2,
            spatial_depth=12,
            temporal_depth=4,
            heads=12,
            channels=1
        )

        # CT-CLIP v2 的特征维度是 768
        self.hidden_size = 768

        if self.vision_tower_path:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            return

        if os.path.exists(self.vision_tower_path):
            print(f"[*] 正在加载 CT-CLIP 预训练权重: {self.vision_tower_path}")
            ckpt = torch.load(self.vision_tower_path, map_location="cpu")

            # 清洗权重字典的前缀
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            vision_state_dict = {k.replace('visual.', ''): v for k, v in state_dict.items() if
                                 'visual' in k or 'transformer' in k}

            try:
                self.vision_tower.load_state_dict(vision_state_dict, strict=True)
                print("[*] CT-CLIP 权重加载成功！")
            except Exception as e:
                print(f"[!] 权重加载过程中出现部分不匹配 (通常是投射层): {e}")

            # --- 核心修复 1：强制隔离为 Float32 ---
            # 视觉塔内部含有 VQ 库，必须强制在 FP32 下运行以避免底层算子精度冲突
            self.vision_tower.float()
            self.is_loaded = True
        else:
            print(f"[!] 未找到路径 {self.vision_tower_path}，视觉塔将保持随机初始化状态运行！")
            self.vision_tower.float()

        # 3. 冻结参数：预训练适配器阶段不需要更新视觉塔权重
        for param in self.vision_tower.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        images 形状: [Batch, 1, 32, 224, 224] (通常来自 LLM 侧，精度可能是 BF16)
        """
        # 1. 记录原始精度 (通常是 BFloat16)
        main_dtype = images.dtype

        # 2. 动态获取当前子显卡设备
        # 在多卡 DDP 环境下， images 已经在对应的 GPU 上，我们需要确保 vision_tower 在同一张卡上
        target_device = images.device

        # 3. --- 核心修复 2：精度转换与对齐 ---
        # 无论外部是什么精度，强制将输入转为 Float32 送入视觉塔
        images_input = images.to(device=target_device, dtype=torch.float32).contiguous()

        # 二次确认权重没有被某些框架偷偷转回 BF16
        if next(self.vision_tower.parameters()).dtype != torch.float32:
            self.vision_tower.float()

        with torch.no_grad():
            # 必须关闭 autocast，否则 amp 会在计算过程中自动把部分算子转回 BF16 导致 VQ 库报错
            with torch.amp.autocast('cuda', enabled=False):
                # 关闭 cuDNN 以绕过某些 3D 卷积的底层版本 Bug
                cudnn_orig = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False

                # 调用底层 CTViT
                image_features = self.vision_tower(
                    images_input,
                    return_encoded_tokens=True
                )

                torch.backends.cudnn.enabled = cudnn_orig

        # 4. 后处理特征维度
        # 将 3D 特征块展平为序列: [B, T, H, W, D] -> [B, T*H*W, D]
        if image_features.ndim == 5:
            image_features = image_features.flatten(1, 3)

        # 5. --- 核心修复 3：精度写回 ---
        # 将输出强制转换回 main_dtype (BF16)，以便无缝对接后续的 Projector 和 LLM
        return image_features.to(dtype=main_dtype)

    @property
    def dummy_feature(self):
        # 辅助函数，用于架构初始化时的占位
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float32)

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype