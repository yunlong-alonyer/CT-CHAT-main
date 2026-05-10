import torch
import torch.nn as nn
import os
from transformer_maskgit.ctvit import CTViT


class CTCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_path, args, **kwargs):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = "ctclip"
        self.vision_tower_path = vision_tower_path

        if self.vision_tower_path is None:
            self.vision_tower_path = getattr(args, 'vision_tower_path', None)

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
        if self.is_loaded: return
        if os.path.exists(self.vision_tower_path):
            print(f"[*] 正在加载 CT-CLIP 预训练权重: {self.vision_tower_path}")
            ckpt = torch.load(self.vision_tower_path, map_location="cpu")
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            vision_state_dict = {k.replace('visual.', ''): v for k, v in state_dict.items() if
                                 'visual' in k or 'transformer' in k}
            self.vision_tower.load_state_dict(vision_state_dict, strict=False)
            print("[*] CT-CLIP 权重加载成功！")

            # --- 隔离区核心 1：强制整个模块到 Float32 并冻结 ---
            self.vision_tower.float()
            self.is_loaded = True
        else:
            print(f"[!] 未找到权重，使用随机初始化！")
            self.vision_tower.float()

        for param in self.vision_tower.parameters():
            param.requires_grad = False

    def forward(self, images):
        # 1. 记录主模型期望的原始精度（BF16）
        main_dtype = images.dtype

        # 2. 检查设备：DeepSpeed 可能会移动 vision_tower
        target_device = next(self.vision_tower.parameters()).device

        # 3. 核心隔离逻辑：强制将输入转为 Float32
        # 即使外部是 BF16，进入视觉塔的必须是 Float32
        images_input = images.to(device=target_device, dtype=torch.float32).contiguous()

        # 4. 强制视觉塔权重保持在 Float32
        # (防止某些 Trainer 在运行过程中又偷偷把模型转回 BF16)
        if next(self.vision_tower.parameters()).dtype != torch.float32:
            self.vision_tower.float()

        with torch.no_grad():
            # 5. 关键：关闭 autocast！
            # 防止 BF16 的全局开关自动把视觉塔内部算子降级回 BF16
            with torch.amp.autocast('cuda', enabled=False):
                # 记录原始 cuDNN 状态并暂时关闭（绕过 3D 卷积 Bug）
                cudnn_orig = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False

                image_features = self.vision_tower(
                    images_input,
                    return_encoded_tokens=True
                )

                torch.backends.cudnn.enabled = cudnn_orig

        # 6. 后处理
        if image_features.ndim == 5:
            image_features = image_features.flatten(1, 3)

        # 7. 核心隔离逻辑：出口处转回 BF16
        # 这样中间的 Projector 拿到的依然是它认得的 BF16，无缝衔接
        return image_features.to(dtype=main_dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float32)

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype