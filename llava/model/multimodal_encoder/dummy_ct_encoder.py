import torch
import torch.nn as nn


class DummyCTVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_loaded = True

        # 假设未来真实的 CT 编码器输出的特征维度是 1024
        # 这个值必须和你 config 里的 mm_hidden_size 一致
        self.hidden_size = getattr(config, 'mm_hidden_size', 1024)

        # 假设未来真实的 3D CT 编码器，对于一个 CT 影像会输出多少个 token
        # 比如经过 3D 卷积下采样后，变成了 4096 个特征块
        self.sequence_length = 4096

    def forward(self, images):
        # images 是 dataloader 传进来的图片/CT张量
        batch_size = images.shape[0] if isinstance(images, torch.Tensor) else len(images)

        # 核心：我们不管输入是什么，直接“伪造”一个输出张量
        # 维度：[Batch, 序列长度, 特征维度] -> [Batch, 4096, 1024]
        device = images.device if isinstance(images, torch.Tensor) else images[0].device
        dtype = self.config.model_dtype if hasattr(self.config, 'model_dtype') else torch.float16

        dummy_features = torch.randn(batch_size, self.sequence_length, self.hidden_size, device=device, dtype=dtype)

        return dummy_features

    @property
    def dummy_feature(self):
        # 提供一个 dummy 属性防报错
        return torch.zeros(1, self.hidden_size)