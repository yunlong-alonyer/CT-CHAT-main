import torch
from types import SimpleNamespace
from llava.model.multimodal_encoder.builder import build_vision_tower

# 1. 伪造一个配置文件
config = SimpleNamespace(
    mm_vision_tower="ctclip",
    vision_tower_path="./CT-CLIP_v2.pt" # 替换为真实路径
)

# 2. 构建视觉塔
print("构建 CT-CLIP 视觉塔...")
vision_tower = build_vision_tower(config)
vision_tower.cuda()

# 3. 伪造一个 3D CT 张量输入 [Batch=1, Channel=1, Depth=32, Height=240, Width=240]
print("生成 Dummy 3D CT Tensor...")
dummy_3d_ct = torch.randn(1, 1, 32, 240, 240).cuda()

# 4. 执行前向传播
print("执行前向传播提取特征...")
try:
    features = vision_tower(dummy_3d_ct)
    print(f"\n[SUCCESS] 前向传播成功！")
    print(f"输出特征的 Shape: {features.shape}")
    # 期望看到类似 [1, N, 768] 的输出，N 是切块的数量 (如 2048)
except Exception as e:
    print(f"\n[FAILED] 前向传播失败: {e}")