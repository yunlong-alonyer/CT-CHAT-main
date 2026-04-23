import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

# 引入 Qwen3-VL 官方模型类
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration

# 引入 LLaVA 的多模态架构基类
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaQwenConfig(AutoConfig):
    model_type = "llava_qwen"


# 多重继承：继承 Qwen3VL 的底层逻辑，同时混入 LLaVA 的多模态处理方法
class LlavaQwenForCausalLM(Qwen3VLForConditionalGeneration, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # 1. 初始化 Qwen3-VL 原生模型
        super(Qwen3VLForConditionalGeneration, self).__init__(config)

        # 2. 对齐 LLaVA 与 Qwen3 的特殊 Token
        # 根据你的 config 文件，Qwen3-VL 使用 151655 作为 image_token_id
        self.config.image_token_id = getattr(config, 'image_token_id', 151655)

        # LLaVA 架构在初始化时，会自动通过 builder.py 往 self.get_model() 中
        # 挂载 vision_tower (CT编码器) 和 mm_projector (你的 AttentionalPooler)

    def get_model(self):
        # LLaVA 的基类需要通过这个方法获取模型主体，以便挂载多模态模块
        return self

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
        # 核心“劫持”逻辑：
        # 如果传入了 images (3D CT 数据) 并且尚未生成 inputs_embeds
        # 则调用 LLaVA 的方法，经过我们的 CT Encoder 和 Attention-Pooler
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
        # 将拼装好 CT Token 的 inputs_embeds 喂给 Qwen3-VL 的底座。
        # 由于我们提供了 inputs_embeds，Qwen 会自动跳过内部的 2D 处理流，
        # 直接进行语言模型的推理。
        # ==========================================================
        return super(Qwen3VLForConditionalGeneration, self).forward(
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
        _inputs = super(Qwen3VLForConditionalGeneration, self).prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


# 注册模型，使其可以通过 AutoModel 方式加载
AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)