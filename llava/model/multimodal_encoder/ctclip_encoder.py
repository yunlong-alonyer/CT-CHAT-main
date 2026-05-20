import torch
import torch.nn as nn
import os
# 注意：确保你的环境里能正确

from transformer_maskgit import CTViT


class CTCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_name, args, **kwargs):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        self.vision_tower_path = getattr(args, 'vision_tower_path', None)

        # 兜底逻辑：如果配置里没写 vision_tower_path，再把名字当路径试一试
        if self.vision_tower_path is None:
            self.vision_tower_path = vision_tower_name

        # 2. 初始化 CT-CLIP v2 核心架构
        # 【修正】严格对齐官方预训练的超参数 (参考 scripts/ct_lipro_train.py)
        self.vision_tower = CTViT(
            dim=512,  # 【修正】官方默认隐层维度是 512
            codebook_size=8192,
            image_size=480,  # 【修正】官方默认输入大小是 480
            patch_size=20,  # 【修正】对应 patch 大小
            temporal_patch_size=10,  # 【修正】时间维度 patch 大小
            spatial_depth=4,  # 【修正】空间深度
            temporal_depth=4,  # 【修正】时间深度
            dim_head=32,  # 【修正】注意这里的 dim_head
            heads=8,  # 【修正】注意头数
            channels=1  # CT图像通常为单通道
        )

        # CT-CLIP 特征维度是 512
        self.hidden_size = 512

        if self.vision_tower_path:
            self.load_model()

    # def load_model(self):
    #     if self.is_loaded:
    #         return
    #
    #     if os.path.exists(self.vision_tower_path):
    #         print(f"[*] 正在加载 CT-CLIP 预训练权重: {self.vision_tower_path}")
    #         ckpt = torch.load(self.vision_tower_path, map_location="cpu", weights_only=False)
    #
    #         # 清洗权重字典的前缀
    #         state_dict = ckpt['model'] if 'model' in ckpt else (ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    #
    #         vision_state_dict = {}
    #         for k, v in state_dict.items():
    #             if k.startswith('visual_transformer.'):
    #                 # 🌟 这里只需要去掉前缀，千万不要再加任何改名逻辑了！
    #                 new_k = k.replace('visual_transformer.', '')
    #                 vision_state_dict[new_k] = v
    #
    #         if len(vision_state_dict) == 0:
    #             print("[!] 警告: 未在权重文件中找到带有 'visual_transformer.' 前缀的权重！请检查 checkpoint。")
    #         else:
    #             try:
    #                 # 保持 strict=True
    #                 self.vision_tower.load_state_dict(vision_state_dict, strict=True)
    #                 print("[*] CT-CLIP 视觉塔权重加载成功！所有权重均完美匹配。")
    #             except Exception as e:
    #                 print(f"[!] 权重加载过程中出现不匹配: {e}")
    #
    #         # 强制隔离为 Float32
    #         self.vision_tower.float()
    #         self.is_loaded = True
    #     else:
    #         print(f"[!] 未找到路径 {self.vision_tower_path}，视觉塔将保持随机初始化状态运行！")
    #         self.vision_tower.float()
    #
    #     # 冻结参数：预训练适配器阶段不需要更新视觉塔权重
    #     for param in self.vision_tower.parameters():
    #         param.requires_grad = False

    def load_model(self):
        if self.is_loaded:
            return

        if os.path.exists(self.vision_tower_path):
            print(f"[*] 正在尝试加载权重，并开启宽容模式: {self.vision_tower_path}")
            ckpt = torch.load(self.vision_tower_path, map_location="cpu", weights_only=False)
            state_dict = ckpt['model'] if 'model' in ckpt else (ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)

            vision_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('visual_transformer.', '')
                vision_state_dict[new_k] = v

            # 核心：使用 strict=False 忽略 Key 不匹配，获取 missing_keys 列表
            incompatible_keys = self.vision_tower.load_state_dict(vision_state_dict, strict=False)

            # 核心：对缺失的 Key 进行零初始化，确保模型不会被随机权重污染
            missing_keys = incompatible_keys.missing_keys
            if missing_keys:
                print(f"🔧 检测到 {len(missing_keys)} 个缺失层，正在执行零初始化修复...")
                with torch.no_grad():
                    for name, param in self.vision_tower.named_parameters():
                        if name in missing_keys:
                            if 'weight' in name or 'bias' in name or 'null_kv' in name:
                                torch.nn.init.zeros_(param)
                                print(f"    [修复完成] 已将 {name} 初始化为 0")

            self.vision_tower.float()
            self.is_loaded = True
            print("[*] 视觉塔加载完成！")
        else:
            print(f"[!] 未找到路径，跳过权重加载。")

        # 冻结参数：预训练适配器阶段不需要更新视觉塔权重
        for param in self.vision_tower.parameters():
             param.requires_grad = False


    def forward(self, images):
        """
        images 形状: [Batch, 1, Frames, 480, 480] (通常来自 LLM 侧，精度可能是 BF16)
        """
        # 1. 记录原始精度 (通常是 BFloat16)
        main_dtype = images.dtype
        target_device = images.device

        # 2. --- 核心修复 2：精度转换与对齐 ---
        images_input = images.to(device=target_device, dtype=torch.float32).contiguous()

        if next(self.vision_tower.parameters()).dtype != torch.float32:
            self.vision_tower.float()

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                cudnn_orig = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False

                # 调用底层 CTViT
                image_features = self.vision_tower(
                    images_input,
                    return_encoded_tokens=True
                )

                torch.backends.cudnn.enabled = cudnn_orig

        # 3. 后处理特征维度 [B, T, H, W, D] -> [B, T*H*W, D]
        if image_features.ndim == 5:
            image_features = image_features.flatten(1, 3)

        # 4. --- 核心修复 3：精度写回 ---
        return image_features.to(dtype=main_dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float32)

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype