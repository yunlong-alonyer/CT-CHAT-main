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

        self.hidden_size = 768

        if self.vision_tower_path:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            return

        if os.path.exists(self.vision_tower_path):
            print(f"[*] 正在加载 CT-CLIP 预训练权重: {self.vision_tower_path}")
            ckpt = torch.load(self.vision_tower_path, map_location="cpu")

            # 清洗权重字典
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            vision_state_dict = {k.replace('visual.', ''): v for k, v in state_dict.items() if
                                 'visual' in k or 'transformer' in k}

            try:
                self.vision_tower.load_state_dict(vision_state_dict, strict=False)
                print("[*] CT-CLIP 权重加载成功！")
            except Exception as e:
                print(f"[!] 权重加载警告: {e}")

            # --- 核心修复点 1：加载后立即强制转为 bfloat16 ---
            # 这一步是为了将第三方库 vector_quantize_pytorch 内部的 Float32 参数转为 BF16
            self.vision_tower.to(torch.bfloat16)
            # ----------------------------------------------

            self.is_loaded = True
        else:
            print(f"[!] 未找到路径 {self.vision_tower_path}，将使用随机初始化权重！")
            self.vision_tower.to(torch.bfloat16) # 即使随机初始化也要对齐精度

        # 冻结视觉塔权重
        for param in self.vision_tower.parameters():
            param.requires_grad = False

    def forward(self, images):
        # 1. 获取主模型期望的精度（通常是 BF16）
        main_dtype = images.dtype

        # 2. 强制将输入转为 Float32，并推送到视觉塔所在的设备
        # 这样能保证进入 CTViT 的所有数据都是 Float32
        images_input = images.to(device=self.device, dtype=torch.float32).contiguous()

        # 3. 运行视觉塔（全程强制在 Float32 环境下）
        with torch.no_grad():
            # 暂时关闭 autocast，防止它自动把算子转回 BF16 导致冲突
            with torch.amp.autocast('cuda', enabled=False):
                # 记录原始 cuDNN 状态，暂时关闭它以绕过 3D 卷积 Bug
                cudnn_orig = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False

                image_features = self.vision_tower(
                    images_input,
                    return_encoded_tokens=True
                )

                torch.backends.cudnn.enabled = cudnn_orig

        # 4. 后处理维度
        if image_features.ndim == 5:
            image_features = image_features.flatten(1, 3)

        # 5. 核心修复点：将输出结果转回主模型的精度（BF16）
        # 这样后续的 Projector 和 LLM 拿到的是它们认得的精度
        return image_features.to(dtype=main_dtype)

    @property
    def dummy_feature(self):
        # 获取当前设备和精度用于生成 dummy tensor
        vt_param = next(self.vision_tower.parameters())
        return torch.zeros(
            1,
            self.hidden_size,
            device=vt_param.device,
            dtype=vt_param.dtype
        )

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype