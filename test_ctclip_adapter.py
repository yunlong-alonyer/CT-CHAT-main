import torch
from types import SimpleNamespace
from transformers import AutoConfig

from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector

# 0. 设定 Qwen3.5 模型路径和 CT-CLIP 路径
QWEN_MODEL_PATH = "./pretrained_models/Qwen-VL"  # 请替换为你实际的本地 Qwen3.5 文件夹名
CT_CLIP_PATH = "./pretrained_models/Qwen-VL/CT-CLIP_v2.pt"  # 请替换为真实的 CT-CLIP 模型路径

# 1. 动态获取 Qwen3.5 的 hidden_size
# Qwen3.5-9B 的 hidden_size 通常是 3584，但最好动态读取
try:
    qwen_config = AutoConfig.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    llm_hidden_size = getattr(qwen_config, "hidden_size", 3584)
    print(f"[INFO] 成功读取 Qwen3.5 配置，LLM hidden_size: {llm_hidden_size}")
except Exception as e:
    print(f"[WARNING] 无法读取本地 Qwen3.5 配置，使用默认隐藏层维度 3584。错误: {e}")
    llm_hidden_size = 3584

# 2. 伪造一个多模态配置
config = SimpleNamespace(
    mm_vision_tower="ctclip",
    vision_tower_path=CT_CLIP_PATH,
    mm_projector_type="coca_pooler",  # 或者 "mlp2x_gelu"，取决于你在 builder.py 里的实现
    mm_hidden_size=768,  # CT-CLIP 编码后的输出维度 (根据实际情况调整)
    hidden_size=llm_hidden_size  # 动态获取的 Qwen3.5 的输入维度
)

# 3. 构建视觉塔 (Vision Tower)
print("\n[1/3] 构建 CT-CLIP 视觉塔...")
vision_tower = build_vision_tower(config)
vision_tower.cuda().eval()

# 4. 构建适配器 (Projector)
print("[2/3] 构建 Projector (适配器)...")
mm_projector = build_vision_projector(config)
mm_projector.cuda().eval()

# 5. 伪造 3D CT 张量输入
print("[3/3] 生成 Dummy 3D CT Tensor...")
dummy_3d_ct = torch.randn(1, 1, 32, 240, 240).cuda()

# 6. 执行端到端前向传播测试
print("\n>>> 开始执行前向传播特征提取与对齐...")
try:
    with torch.no_grad():
        # A. 视觉塔提取特征
        vision_features = vision_tower(dummy_3d_ct)
        print(f"  [✓] 视觉塔前向传播成功！提取特征 Shape: {vision_features.shape}")

        # LLaVA 架构通常将图像特征打包成 list 或直接输入投影层
        if isinstance(vision_features, list):
            vision_features = vision_features[0]

        # B. 适配器特征对齐
        projected_features = mm_projector(vision_features)
        print(f"  [✓] 适配器对齐成功！")
        print(f"  [✓] 最终特征 Shape: {projected_features.shape}")

        # C. 验证维度
        if projected_features.shape[-1] == llm_hidden_size:
            print(
                f"\n[SUCCESS] 测试完美跑通！适配器输出维度 {projected_features.shape[-1]} 与 Qwen3.5 输入维度 {llm_hidden_size} 完全匹配！")
        else:
            print(f"\n[WARNING] 跑通了，但维度不匹配！适配器输出: {projected_features.shape[-1]}，期望: {llm_hidden_size}")

except Exception as e:
    print(f"\n[FAILED] 前向传播失败，请检查报错详情: {e}")