import os
from .clip_encoder import CLIPVisionTower

from .ctclip_encoder import CTCLIPVisionTower # <--- 新增导入


def build_vision_tower(vision_tower_cfg, **kwargs):
    # 获取 vision_tower 的配置字符串
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    # --- 核心修改开始 ---
    # 如果路径指向具体的 .pt 文件，或者包含 'ctclip' 关键词，则判定为 CT-CLIP 视觉塔
    if vision_tower.endswith('.pt') or 'ct-clip' in vision_tower.lower() or 'ctclip' in vision_tower.lower():
        from .ctclip_encoder import CTCLIPVisionTower
        return CTCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # --- 核心修改结束 ---

    # 以下是原有的 CLIP 或其他编码器的判断逻辑
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        from .clip_encoder import CLIPVisionTower
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')