import torch

# 1. 确认视觉塔输出维度
ckpt = torch.load("./checkpoint/CT-CLIP_v2.pt", map_location="cpu", weights_only=False)
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# 找输出投影层，直接看编码器最终输出维度
for k, v in state_dict.items():
    if 'visual_transformer' in k and 'norm_out' in k:
        print(f"[视觉塔 norm_out] {k}: {v.shape}")
    if k == 'to_visual_latent.weight' or k == 'to_visual_latent_extra.weight':
        print(f"[视觉投影] {k}: {v.shape}")

# 2. 确认 projector checkpoint 的维度
proj_ckpt = torch.load("/mnt/huali/checkpoint_projector/epoch_2/mm_projector.bin",
                        map_location="cpu", weights_only=False)
print("\n[Projector 权重 shapes]")
for k, v in proj_ckpt.items():
    print(f"  {k}: {v.shape}")