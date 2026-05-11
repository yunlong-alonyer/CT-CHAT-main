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
        if self.is_loaded:
            return

        if os.path.exists(self.vision_tower_path):
            print(f"[*] 正在加载 CT-CLIP 预训练权重: {self.vision_tower_path}")
            ckpt = torch.load(self.vision_tower_path, map_location="cpu")

            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            vision_state_dict = {k.replace('visual.', ''): v for k, v in state_dict.items() if
                                 'visual' in k or 'transformer' in k}

            try:
                self.vision_tower.load_state_dict(vision_state_dict, strict=False)
                print("[*] CT-CLIP 权重加载成功！")
            except Exception as e:
                print(f"[!] 权重加载过程中出现部分不匹配: {e}")

            self.vision_tower.float()
            self.is_loaded = True
        else:
            print(f"[!] 未找到路径 {self.vision_tower_path}，视觉塔将保持随机初始化状态运行！")
            self.vision_tower.float()

        for param in self.vision_tower.parameters():
            param.requires_grad = False

    def forward(self, images):
        main_dtype = images.dtype
        target_device = images.device

        # 完全保留你预训练时的逻辑：强制输入为 Float32
        images_input = images.to(device=target_device, dtype=torch.float32).contiguous()

        # 【完美兼容的核心修改】
        # 这行代码在 ZeRO-2 预训练时是极速的空操作 (No-op)；
        # 在 ZeRO-3 微调时，它会强行把遗留在 CPU 上的 VQ 密码本 (Buffer) 抓进 GPU，并确保是 FP32。
        self.vision_tower.to(device=target_device, dtype=torch.float32)

        with torch.no_grad():
            # 完全保留你预训练时的逻辑：关闭 autocast 防止内部算子被切回 BF16
            with torch.amp.autocast('cuda', enabled=False):
                cudnn_orig = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False

                image_features = self.vision_tower(
                    images_input,
                    return_encoded_tokens=True
                )

                torch.backends.cudnn.enabled = cudnn_orig

        if image_features.ndim == 5:
            image_features = image_features.flatten(1, 3)

        return image_features.to(dtype=main_dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype