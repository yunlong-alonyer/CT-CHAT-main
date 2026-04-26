import torch
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoConfig
# 引入你之前修改好的自定义 Qwen 类
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# 1. 路径配置 (按照你的真实路径)
QWEN_DIR = "./pretrained_models/Qwen-VL"
CT_CLIP_PATH = "./pretrained_models/Qwen-VL/CT-CLIP_v2.pt"

print(f"[*] 正在加载 Qwen3.5 配置与 Tokenizer: {QWEN_DIR}")
tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
# 注意：Qwen3.5 比较新，一定要确保使用了正确的 config 类
raw_config = AutoConfig.from_pretrained(QWEN_DIR, trust_remote_code=True)

# 2. 构造多模态模型配置
# 我们需要将 LLaVA 相关的参数注入到 Qwen 的 config 中
multimodal_cfg = {
    "mm_vision_tower": "ctclip",
    "vision_tower_path": CT_CLIP_PATH,
    "mm_projector_type": "coca_pooler",
    "mm_hidden_size": 768,
    "hidden_size": 4096,  # 对应你 config.json 中的 text_config.hidden_size
    "image_token_id": 248056,  # 对应你 config.json 中的值
}

for k, v in multimodal_cfg.items():
    setattr(raw_config, k, v)

# 3. 实例化模型 (为了测试流程，我们先不加载全量 LLM 权重，或者以半精度加载)
print("[*] 正在初始化 LlavaQwenForCausalLM 模型架构...")
# 如果显存足够，直接使用 from_pretrained；如果只是测逻辑，可以先不 load 权重
# 这里使用 from_pretrained 加载你 safetensors 中的 Qwen 权重
model = LlavaQwenForCausalLM.from_pretrained(
    QWEN_DIR,
    config=raw_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# 手动推送到 GPU
model = model.cuda()

# 4. 手动构建并挂载视觉模块 (因为是测试脚本)
print("[*] 挂载视觉塔与适配器...")
model.get_model().vision_tower = build_vision_tower(raw_config)
model.get_model().mm_projector = build_vision_projector(raw_config)

# 先把整个模型转为半精度 FP16，并放到显卡
model.to(dtype=torch.float16, device="cuda")

# 【核心修改】：把第三方库的视觉塔单独抽出来，强行转回 FP32，迎合它内部的硬编码！
model.get_model().vision_tower.to(torch.float32)

model.eval()

# 5. 准备测试输入
# 构建 Prompt: <image>\n请根据这张CT图像生成一份诊断报告。
prompt = f"{DEFAULT_IMAGE_TOKEN}\nPlease provide a detailed diagnostic report for this 3D CT scan."
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

# 伪造 3D CT 数据 [Batch=1, C=1, D=32, H=240, W=240]
images = torch.randn(1, 1, 32, 240, 240).half().cuda()

# 6. 执行推理
print("\n>>> 开始端到端前向传播并生成文本...")
try:
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=128,
            use_cache=True
        )

    # ... 上面是 model.generate ...

    # 获取输入 prompt 的长度
    input_token_len = input_ids.shape[1]

    # 截取模型 [新生成] 的输出部分，避开带有 -200 的 prompt 区域
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] 有 {n_diff_input_output} 个输出 token 和输入不一致。")

    # 提取新生成的 token
    new_tokens = output_ids[0][input_token_len:].tolist()

    # 保险起见，再过滤掉任何可能的负数 Token (如 -200)
    valid_tokens = [t for t in new_tokens if t >= 0]

    # 解码
    response = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()

    print("\n" + "=" * 50)
    print("[Qwen3.5 输出结果]:")
    print(response)
    print("=" * 50)
    print("\n[SUCCESS] 流程全部跑通！从 3D CT 输入到文本输出闭环完成。")

except Exception as e:
    print(f"\n[FAILED] 推理流出错: {e}")
    import traceback

    traceback.print_exc()