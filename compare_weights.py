import torch


def extract_state_dict(checkpoint):
    """
    尝试智能提取状态字典，兼容不同的保存格式
    （例如有些保存在 ckpt['model'] 下，有些在 ckpt['state_dict'] 下）
    """
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            return checkpoint['model']
        elif 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            # 如果没有上述 key，假设它本身就是一个直接的 state_dict
            return checkpoint
    return checkpoint


def compare_encoders(path_a, path_b, prefix_a="", prefix_b="", rtol=1e-5, atol=1e-8):
    """
    对比两个 PyTorch 模型权重文件
    :param path_a: 模型A的文件路径
    :param path_b: 模型B的文件路径
    :param prefix_a: 过滤模型A权重键名的前缀 (如 'visual_transformer.')
    :param prefix_b: 过滤模型B权重键名的前缀
    :param rtol: 相对误差容忍度 (针对 float 精度问题)
    :param atol: 绝对误差容忍度
    """
    print(f"[*] 正在加载模型 A: {path_a}")
    try:
        ckpt_a = torch.load(path_a, map_location='cpu', weights_only=False)
        state_a = extract_state_dict(ckpt_a)
    except Exception as e:
        print(f"[!] 模型 A 加载失败: {e}")
        return

    print(f"[*] 正在加载模型 B: {path_b}")
    try:
        ckpt_b = torch.load(path_b, map_location='cpu', weights_only=False)
        state_b = extract_state_dict(ckpt_b)
    except Exception as e:
        print(f"[!] 模型 B 加载失败: {e}")
        return

    # 1. 清洗前缀（针对多模态模型中常出现的嵌套前缀）
    if prefix_a:
        state_a = {k.replace(prefix_a, ""): v for k, v in state_a.items()}
    if prefix_b:
        state_b = {k.replace(prefix_b, ""): v for k, v in state_b.items()}

    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())

    # 2. 对比键值结构 (Key names)
    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a
    common_keys = keys_a & keys_b

    print("\n" + "=" * 50)
    print("📊 权重结构分析:")
    if only_in_a:
        print(f"  [⚠️] 模型A独有的参数: {len(only_in_a)} 个 (例如: {list(only_in_a)[:2]}...)")
    if only_in_b:
        print(f"  [⚠️] 模型B独有的参数: {len(only_in_b)} 个 (例如: {list(only_in_b)[:2]}...)")

    if not common_keys:
        print("\n[❌] 致命错误：两个模型没有任何匹配的参数名！请检查代码中的 prefix 是否设置正确。")
        return

    # 3. 对比共同参数的数值和形状
    print(f"\n🔍 开始对比 {len(common_keys)} 个共有参数的内部数值...")
    shape_mismatch = []
    value_mismatch = []
    exact_match_count = 0

    for key in common_keys:
        tensor_a = state_a[key]
        tensor_b = state_b[key]

        # 检查是否为 Tensor
        if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
            if tensor_a == tensor_b:
                exact_match_count += 1
            continue

        # 检查形状
        if tensor_a.shape != tensor_b.shape:
            shape_mismatch.append((key, tensor_a.shape, tensor_b.shape))
            continue

        # 将精度对齐为 float32 进行数值对比，避免 fp16/bf16 带来的底层误差误报
        tensor_a_f32 = tensor_a.float()
        tensor_b_f32 = tensor_b.float()

        # 使用 torch.allclose 允许合理的浮点误差
        if torch.allclose(tensor_a_f32, tensor_b_f32, rtol=rtol, atol=atol):
            exact_match_count += 1
        else:
            # 计算最大绝对差值以供排查
            max_diff = torch.max(torch.abs(tensor_a_f32 - tensor_b_f32)).item()
            value_mismatch.append((key, max_diff))

    # 4. 输出最终报告
    print("\n" + "=" * 50)
    print("📋 对比最终报告:")
    print(f"  ✅ 完全一致的层数: {exact_match_count} / {len(common_keys)}")

    if shape_mismatch:
        print(f"\n  📐 形状不一致的层数: {len(shape_mismatch)}")
        for k, sa, sb in shape_mismatch[:5]:  # 只展示前 5 个
            print(f"     - {k}: A {sa}  VS  B {sb}")

    if value_mismatch:
        print(f"\n  ⚠️ 数值不一致的层数: {len(value_mismatch)}")
        for k, diff in value_mismatch[:5]:
            print(f"     - {k}: 最大绝对误差 = {diff:.6f}")

    # 判定结论
    if exact_match_count == len(keys_a) == len(keys_b):
        print("\n🎉 结论：两个编码器的权重【完美匹配，内部完全相同】！")
    elif exact_match_count == len(common_keys) and (only_in_a or only_in_b):
        print("\n⚠️ 结论：共同部分的权重完全一致，但【存在结构差异 (参数层多或少)】。")
    else:
        print("\n❌ 结论：两个编码器的权重【不相同】。")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # 示例用法
    path1 = "./checkpoint/CT-CLIP_v2.pt"  # 替换为你的真实路径
    path2 = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500_full.pt"

    # 如果你对比的是官方 CT-CLIP 和 你 LLaVA 里加载过并保存下来的权重
    # LLaVA 里的权重通常没有前缀，而官方的可能带有 'visual_transformer.'
    compare_encoders(
        path_a=path1,
        path_b=path2,
        prefix_a="visual_transformer.",  # 如果 path1 带有前缀，将其剥离
        prefix_b=""
    )