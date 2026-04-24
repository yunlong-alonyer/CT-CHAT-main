import torch
import torch.nn as nn
import os
# 直接调用刚才安装的 CT-CLIP 底层模型
from transformer_maskgit.ctvit import CTViT


class CTClipVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_loaded = False

        # 从配置中读取权重路径
        self.vision_tower_name = getattr(config, 'mm_vision_tower', 'ctclip')
        self.vision_tower_path = getattr(config, 'vision_tower_path', None)

        # 初始化 CT-CLIP 的 3D ViT
        # 注意：这里的 dim, depth, heads 必须与你训练 CT-CLIP 时的配置一致
        self.vision_tower = CTViT(
            dim=768,
            depth=12,
            heads=12,
            image_size=240,  # 根据实际 CT-CLIP 的输入尺寸调整
            patch_size=16,
            num_channels=1
        )

        self.hidden_size = 768  # 这是 CTViT 输出的维度，对应 config.mm_hidden_size

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
            # 获取 3D 特征，CTViT 输出维度通常为 [Batch, Sequence_Length, Hidden_Dim]
            image_features = self.vision_tower(images)

            # 如果 CTViT 输出包含了 cls_token，通常需要截取掉第一个 token: image_features[:, 1:, :]
            # 具体取决于 CTViT 的 forward 实现
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.vision_tower.device, dtype=self.vision_tower.dtype)