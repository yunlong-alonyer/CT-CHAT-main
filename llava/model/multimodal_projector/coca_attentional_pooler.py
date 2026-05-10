import logging
from typing import Callable, Optional, Sequence, Tuple
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

# 假设你的项目中使用 Registor 或者是从 builder 引入的
from .builder import Registor


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = nn.LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))

        dim_head = d_model // n_head
        self.scale = dim_head ** -0.5
        self.heads = n_head
        inner_dim = dim_head * n_head

        # 核心修改点：这里的 context_dim 必须等于视觉塔的输出维度 (768)
        self.ln_k = norm_layer(context_dim)
        self.ln_q = norm_layer(d_model)

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # 兼容处理：如果是 [B, 1, N, D] 则保持，如果是 [B, N, D] 则增加维度
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        # 针对你报错的 [1, 1, 1568, 768]，直接进入后续逻辑
        q = repeat(self.query, 'n d -> b m n d', b=x.shape[0], m=x.shape[1])

        x = self.ln_k(x)  # 这里现在会正确接收 768 维度的输入
        q = self.ln_q(q)
        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(q)

        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)

        q = q * self.scale

        # attention 计算
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out).squeeze(dim=1)


# 使用 Registor 注册，确保 --mm_projector_type coca_pooler 能找到这个类
@Registor.register("coca_pooler")
class AttentionalPoolProjector(nn.Module):
    def __init__(
            self,
            config,  # 核心修改：接收 config 对象，而不是零散参数
            projector=None,
            n_head=8,
            n_queries=256,
            norm_layer: Callable = nn.LayerNorm):
        super().__init__()

        # 从 config 中动态读取维度
        # embed_dim 对应 LLM 的维度 (Qwen 是 4096)
        # context_dim 对应 视觉塔 的维度 (CT-CLIP v2 是 768)
        embed_dim = getattr(config, "hidden_size", 4096)
        context_dim = getattr(config, "mm_hidden_size", 768)

        self.attn_pool = AttentionalPooler(d_model=embed_dim,
                                           context_dim=context_dim,
                                           n_head=n_head,
                                           n_queries=n_queries)
        self.ln = norm_layer(embed_dim)
        self.proj = projector if projector else nn.Identity()

    def forward(self, x: torch.Tensor):
        # x shape: [B, 1, 1568, 768]
        tokens = self.attn_pool(x)
        tokens = self.ln(tokens)
        tokens = self.proj(tokens)
        return tokens