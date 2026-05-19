import torch
ckpt = torch.load("./checkpoint/CT-CLIP_v2.pt", map_location="cpu", weights_only=False)
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# 只看视觉塔相关的 key 和 shape
for k, v in state_dict.items():
    if 'visual_transformer' in k and any(x in k for x in ['patch_emb', 'to_pixels', 'pos']):
        print(f"{k}: {v.shape}")

# 如果 checkpoint 里有 config 或 hparams
if 'hyper_parameters' in ckpt:
    print("\nhyper_parameters:", ckpt['hyper_parameters'])
if 'config' in ckpt:
    print("\nconfig:", ckpt['config'])