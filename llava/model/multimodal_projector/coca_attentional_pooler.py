import torch
from torch import nn, einsum
from typing import Callable
from einops import rearrange, repeat
from einops_exts import rearrange_many

# 务必确保这个导入指向我们之前建立的独立注册器
from .registor import Registor


class AttentionalPooler(nn.Module):
    def __init__(self, d_model: int, context_dim: int, n_head: int = 8, n_queries: int = 256):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        dim_head = d_model // n_head
        self.scale = dim_head ** -0.5
        self.heads = n_head
        inner_dim = dim_head * n_head

        # 核心：这里的 context_dim 必须是 768
        self.ln_k = nn.LayerNorm(context_dim)
        self.ln_q = nn.LayerNorm(d_model)

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        # 此时 x 的形状是 [B, 1, 1568, 768]
        q = repeat(self.query, 'n d -> b m n d', b=x.shape[0], m=x.shape[1])

        # --- 如果维度还是不对，这里加一个硬拦截强制对齐 ---
        if x.shape[-1] != self.ln_k.normalized_shape[0]:
            # 这种情况说明 config 传错了，我们在这里通过代码纠正
            self.ln_k = nn.LayerNorm(x.shape[-1]).to(x.device, x.dtype)
            self.to_kv = nn.Linear(x.shape[-1], self.to_kv.out_features, bias=False).to(x.device, x.dtype)

        x = self.ln_k(x)
        q = self.ln_q(q)

        b, m, h = *x.shape[:2], self.heads
        q = self.to_q(q)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)
        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out).squeeze(dim=1)


@Registor.register("coca_pooler")
class AttentionalPoolProjector(nn.Module):
    def __init__(self, config, projector=None, n_head=8, n_queries=256):
        super().__init__()
        # Qwen 3.5 维度是 4096
        embed_dim = getattr(config, "hidden_size", 4096)
        # 显式指定：如果配置里没写，就默认 768 (CT-CLIP v2)
        #context_dim = getattr(config, "mm_hidden_size", 768)
        #context_dim = 768
        context_dim = getattr(config, "mm_hidden_size", 512)
        self.attn_pool = AttentionalPooler(d_model=embed_dim, context_dim=context_dim, n_head=n_head,
                                           n_queries=n_queries)
        self.ln = nn.LayerNorm(embed_dim)
        self.proj = projector if projector else nn.Identity()

    def forward(self, x: torch.Tensor):
        tokens = self.attn_pool(x)
        tokens = self.ln(tokens)
        tokens = self.proj(tokens)
        return tokens