import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# =========================================================================
# 1. 动态导入 Qwen3_5ForConditionalGeneration
# 因为 Qwen3.5 是非常新的模型 (transformers_version: 4.57.0.dev0)，
# 原生 transformers 可能还不包含它。如果直接 import 报错，你需要从本地
# pretrained_models/Qwen3.5-9B/modeling_qwen3_5.py 中导入。
# =========================================================================
try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    import sys
    import os

    # 假设你的 Qwen3.5 权重在 pretrained_models/Qwen3.5-9B 下
    sys.path.append(os.path.abspath("../../../pretrained_models/Qwen-VL"))
    try:
        from modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    except ImportError:
        print(
            "[Error] 无法找到 Qwen3_5ForConditionalGeneration 类，请确保 transformers 为最新版或包含本地 modeling_qwen3_5.py")


class LlavaQwenConfig(AutoConfig):
    model_type = "llava_qwen"


# 2. 多重继承：继承正确的 Qwen3_5 基类
class LlavaQwenForCausalLM(Qwen3_5ForConditionalGeneration, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # 初始化 Qwen3.5 原生模型
        super().__init__(config)

        # 3. 对齐 LLaVA 与 Qwen3.5 的特殊 Token
        # 根据 config.json，image_token_id 为 248056
        self.config.image_token_id = getattr(config, 'image_token_id', 248056)

        # 4. 对齐隐藏层维度 (极其重要)
        # Qwen3.5 将 LLM 的参数放在了 text_config 里。
        # LLaVA 的 Projector 需要读取 self.config.hidden_size 来确定输出维度 (4096)
        if hasattr(config, 'text_config'):
            self.config.hidden_size = config.text_config.hidden_size
        else:
            self.config.hidden_size = getattr(config, 'hidden_size', 4096)

        # 初始化时 builder 会挂载 vision_tower 和 mm_projector

    def get_model(self):
        # LLaVA 架构需要通过这里获取底座模型进行特征融合
        return self


    def get_vision_tower(self):
        # 显式获取挂载在模型上的 vision_tower，打破递归循环
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ):
        # ==========================================================
        # 劫持逻辑：如果有 3D CT (images) 且没有 inputs_embeds，
        # 则调用 LLaVA 的 prepare 方法，通过 CT-CLIP 和 Projector 生成 Embeddings
        # ==========================================================
        if inputs_embeds is None and images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        # ==========================================================
        # 将拼装好 CT Token 的 inputs_embeds 喂给 Qwen3.5 的底座
        # ==========================================================
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


# 注册模型
AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)