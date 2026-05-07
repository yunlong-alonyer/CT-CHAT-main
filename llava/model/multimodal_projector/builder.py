import torch
import torch.nn as nn
import re
from .coca_attentional_pooler import AttentionalPoolProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    # =====================================================================
    # 核心修改 1：动态获取目标语言模型的隐藏层维度
    # 兼容 Qwen3-VL 嵌套的 text_config 结构 (5120)，以及 LLaVA 原生的结构。
    # =====================================================================
    if hasattr(config, 'text_config'):
        target_hidden_size = config.text_config.hidden_size
    else:
        target_hidden_size = getattr(config, 'hidden_size', 5120)

    # =====================================================================
    # 核心修改 2：新增 3D CT 影像 Attention-Pooling 分支 (保持架构解耦)
    # =====================================================================
    if projector_type in ['ct_qwen_pooler', 'coca_pooler']:
        # 构建两层 MLP，负责将 CT 编码器的维度升/降维到 Qwen 的 5120 维
        mlp_projector = nn.Sequential(
            nn.Linear(config.mm_hidden_size, target_hidden_size),
            nn.GELU(),
            nn.Linear(target_hidden_size, target_hidden_size)
        )

        # 将构建好的 MLP 作为 projector 传入池化器，完成无缝对接
        return AttentionalPoolProjector(
            embed_dim=config.mm_hidden_size,
            context_dim=getattr(config, 'mm_context_size', config.mm_hidden_size),
            projector=mlp_projector,
            n_queries=256
        )

    # =====================================================================
    # 原版保留逻辑：将所有 config.hidden_size 替换为 target_hidden_size
    # =====================================================================
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, target_hidden_size)

    if projector_type.startswith('attn_pool'):
        mlp_projector = projector_type.split('+')[1]
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mlp_projector)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, target_hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(target_hidden_size, target_hidden_size))
            projector = nn.Sequential(*modules)
        else:
            projector = nn.Linear(config.mm_hidden_size, target_hidden_size)

        mm_projector = AttentionalPoolProjector(embed_dim=config.mm_hidden_size,
                                                context_dim=config.mm_context_size,
                                                projector=projector)
        return mm_projector

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, target_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(target_hidden_size, target_hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')