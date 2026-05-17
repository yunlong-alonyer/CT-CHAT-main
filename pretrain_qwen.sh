# 将当前工作目录加入 Python 寻址路径，解决 llava 模块找不到的问题
export PYTHONPATH="$PWD:$PYTHONPATH"

# 指定使用的 GPU 单卡跑预训练1
#export CUDA_VISIBLE_DEVICES=0

deepspeed --num_gpus 8 llava/train/train_mem.py \
    --deepspeed ./zero2.json \
    --model_name_or_path /home/huali/model/Qwen3.5-9B \
    --version plain \
    --data_path ./dataset_llava_format_10000.json \
    --image_folder /mnt/huali/ct_dataset_10000/pretrain_processed_train_data \
    --vision_tower /mnt/huali/ct_dataset_10000/output/CTClip_step_21000_full.pt \
    --mm_projector_type coca_pooler \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_hidden_size 768 \
    --bf16 True \
    --output_dir ./checkpoints/qwen-ct-pretrain_10000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none
