import os
from .clip_encoder import CLIPVisionTower

from .ctclip_encoder import CTClipVisionTower  # <--- 新增导入


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower.startswith("openai") or vision_tower.startswith("clip"):
        return CLIPVisionTower(vision_tower_cfg, **kwargs)

    elif "ctclip" in vision_tower.lower():  # <--- 新增分支
        return CTClipVisionTower(vision_tower_cfg)

    raise ValueError(f'Unknown vision tower: {vision_tower}')