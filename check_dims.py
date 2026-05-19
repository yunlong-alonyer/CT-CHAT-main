import torch

# 替换为你实际的权重路径
checkpoint_path = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500_full.pt"

print(f"[*] 正在读取权重文件: {checkpoint_path}")
# 这里的 map_location="cpu" 非常重要，避免在没有显卡的环境下报错
ckpt = torch.load(checkpoint_path, map_location="cpu")

# 如果你的权重文件里包含 'state_dict' 键，就取它
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# 打印前 5 个参数的形状，让你一眼看出维度
print("\n[+] 权重文件中的参数形状示例：")
for i, (key, value) in enumerate(state_dict.items()):
    if i >= 5: break
    print(f"层名称: {key:50} | 形状: {list(value.shape)}")

# 如果你想精确查找某一层（例如编码器的第一层 patch embedding）
# 可以搜索包含 patch_emb 的关键字
print("\n[+] 搜索 'to_patch_emb' 相关层维度：")
for key, value in state_dict.items():
    if 'to_patch_emb' in key:
        print(f"找到关键层: {key} | 形状: {list(value.shape)}")