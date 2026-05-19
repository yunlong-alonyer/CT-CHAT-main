import torch
from transformer_maskgit.ctvit import CTViT

model = CTViT(
    dim=512, codebook_size=8192, image_size=240,
    patch_size=20, temporal_patch_size=10,
    spatial_depth=4, temporal_depth=4,
    heads=8, dim_head=32, channels=1
)

ckpt = torch.load("./checkpoint/CT-CLIP_v2.pt", map_location="cpu", weights_only=False)
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
vision_state_dict = {k.replace('visual_transformer.', ''): v
                     for k, v in state_dict.items() if k.startswith('visual_transformer.')}

msg = model.load_state_dict(vision_state_dict, strict=False)

print("=== Missing (随机初始化) ===")
for k in msg.missing_keys:
    print(f"  {k}")

print("\n=== Unexpected (丢弃) ===")
for k in msg.unexpected_keys:
    print(f"  {k}")

# 跑一个前向验证编码器能正常工作
print("\n=== 前向传播验证 ===")
model.eval()
dummy = torch.zeros(1, 1, 30, 240, 240)  # [B, C, D, H, W]
with torch.no_grad():
    out = model(dummy, return_encoded_tokens=True)
print(f"输入: {dummy.shape}")
print(f"输出: {out.shape}")
print("编码器正常！")