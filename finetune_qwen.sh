#!/bin/bash

export PYTHONPATH="$PWD:$PYTHONPATH"

# 推荐使用 zero3.json 以应对 Qwen 较大的显存占用
deepspeed --num_gpus 8 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./zero2.json \
    --model_name_or_path /home/huali/model/Qwen3.5-9B \
    --version qwen \
    --data_path finetune_data_qwen.json \
    --image_folder ../CT-CLIP-main/dataset/pretrain_processed_train_data \
    --vision_tower /home/huali/code/CT-CHAT-main/checkpoint/CT-CLIP_v2.pt \
    --mm_projector_type coca_pooler \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pretrain_mm_mlp_adapter ./checkpoints/qwen-ct-pretrain/mm_projector.bin \
    --mm_hidden_size 768 \
    --bf16 True \
    --output_dir ./checkpoints/qwen-ct-finetune-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none