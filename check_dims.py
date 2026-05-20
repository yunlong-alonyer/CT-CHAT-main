import torch

# 1. 加载同学的权重
ckpt_path = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500_full.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 2. 提取 state_dict
state_dict = ckpt['model'] if 'model' in ckpt else (ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)

# 3. 过滤并打印出缺失的那个 Block 到底有什么参数
print("=== 正在检查同学权重中 enc_spatial_transformer.layers.0 的真实参数 ===")
for k in state_dict.keys():
    if 'enc_spatial_transformer.layers.0' in k:
        print(k)