import torch

# 🚨 将这里的路径替换为你刚才生成的最终单独权重文件的实际路径
weight_path = "./checkpoints/test2/checkpoint-26/mm_projector.bin"


def check_dims():
    print(f"[*] 正在读取纯净版适配器权重: {weight_path}\n")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    print(f"{'=' * 60}")
    print(f"{'网络层名称 (Layer Key)':<45} | {'维度 (Shape)'}")
    print(f"{'-' * 60}")

    for k, v in state_dict.items():
        # 将 shape 转换为易读的列表格式
        shape_str = str(list(v.shape))
        print(f"{k:<45} | {shape_str}")

    print(f"{'=' * 60}")

    # 自动进行关键维度体检
    print("\n[体检报告]:")

    # 检查 1: 输入端是否对齐 CT-CLIP (512维)
    # 注意力池化器的 K/V 线性层输入应该是 512
    if "attn_pool.to_kv.weight" in state_dict:
        to_kv_shape = state_dict["attn_pool.to_kv.weight"].shape
        if to_kv_shape[1] == 512:
            print("✅ 输入端维度正确: 完美适配 CT-CLIP 的 512 维特征输入！")
        else:
            print(f"❌ 输入端维度错误: 期望输入 512，但当前为 {to_kv_shape[1]}")

    # 检查 2: 输出端是否对齐 Qwen3.5 (4096维)
    # MLP 的第一层输入应该是 512，输出（或是最终层输出）应当参与 4096 映射
    if "proj.0.weight" in state_dict:
        proj_0_shape = state_dict["proj.0.weight"].shape
        if proj_0_shape[1] == 512 and proj_0_shape[0] == 4096:
            print("✅ 桥接端维度正确: 完美将 512 维特征升维至 LLM 的 4096 维！")
        else:
            print(f"⚠️ 请人工核对 proj.0.weight 的维度: {proj_0_shape}")


if __name__ == "__main__":
    check_dims()