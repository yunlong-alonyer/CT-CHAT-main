import torch
import torch.nn as nn
import os
# 直接调用刚才安装的 CT-CLIP 底层模型
from transformer_maskgit.ctvit import CTViT


class CTCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_path, args, **kwargs):
        super().__init__()
        self.is_loaded = False

        # 1. 直接使用 builder 传进来的路径
        self.vision_tower_name = "ctclip"
        self.vision_tower_path = vision_tower_path

        # 如果 builder 没传路径过来，尝试从 args 配置里读（兜底逻辑）
        if self.vision_tower_path is None:
            self.vision_tower_path = getattr(args, 'vision_tower_path', None)

        # 2. 初始化 CT-CLIP 的 3D ViT
        # (保持原有的 CTViT 配置不变)
        self.vision_tower = CTViT(
            dim=768,
            codebook_size=8192,
            image_size=240,
            patch_size=16,
            temporal_patch_size=2,
            spatial_depth=12,
            temporal_depth=4,
            heads=12,
            channels=1
        )

        self.hidden_size = 768

        if self.vision_tower_path:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            return

        if os.path.exists(self.vision_tower_path):
            print(f"[*] 正在加载 CT-CLIP 预训练权重: {self.vision_tower_path}")
            ckpt = torch.load(self.vision_tower_path, map_location="cpu")

            # CT-CLIP 的权重字典前缀可能需要清洗（剥离对比学习层的权重，只保留 ViT）
            # 这里假设保存的权重前缀为 'visual.' 或者直接是模型参数
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            vision_state_dict = {k.replace('visual.', ''): v for k, v in state_dict.items() if
                                 'visual' in k or 'transformer' in k}

            try:
                self.vision_tower.load_state_dict(vision_state_dict, strict=False)
                print("[*] CT-CLIP 权重加载成功！")
            except Exception as e:
                print(f"[!] 权重加载警告 (可忽略未匹配的投射层): {e}")

            self.is_loaded = True
        else:
            print(f"[!] 未找到路径 {self.vision_tower_path}，将使用随机初始化权重！")

        # 冻结视觉塔权重，只训练后续的 Attentional Pooler 和 LLM
        for param in self.vision_tower.parameters():
            param.requires_grad = False

    def forward(self, images):
        # 接收形状: [Batch, Channel, Depth, Height, Width]
        with torch.no_grad():
            # 1. 动态获取当前视觉塔的精度 (这里已被我们设置成了 FP32)
            vt_dtype = next(self.vision_tower.parameters()).dtype

            # 2. 将输入的图像强行对齐到视觉塔的精度 (FP16 -> FP32)
            images_input = images.to(vt_dtype)

            # 3. 提取特征 (在 FP32 下安全运行，不会报硬编码 float 错误)
            image_features = self.vision_tower(
                images_input,
                return_encoded_tokens=True
            )

            if image_features.ndim == 5:
                image_features = image_features.flatten(1, 3)

             # 4. 将输出特征强行转回外部大模型的精度 (FP32 -> FP16)，无缝交接给 Projector
            # fallback 取 images 的原精度即可 (即最开始传进来的 FP16)
            return image_features.to(images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.vision_tower.device, dtype=self.vision_tower.dtype)