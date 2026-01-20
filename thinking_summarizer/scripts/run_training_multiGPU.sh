#!/bin/bash

# Thinking Summarizer 多卡训练启动脚本
# 使用4张A800 GPU进行分布式训练

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 使用torchrun启动多卡训练
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  scripts/train_summarizer.py \
  --model-path ${PROJECT_ROOT}/Qwen2.5-3B-Instruct \
  --train-file ./data/training_dataset/train.json \
  --val-file ./data/training_dataset/val.json \
  --output-dir ./outputs/thinking_summarizer \
  --max-length 2048 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --num-epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 2 \
  --learning-rate 3e-4 \
  --warmup-ratio 0.1 \
  --save-steps 50 \
  --logging-steps 10

# 说明：
# - nproc_per_node=4: 使用4张GPU
# - batch-size=2: 每张卡batch size为2
# - gradient-accumulation=2: 梯度累积2步
# - 有效batch size = 4 GPUs × 2 batch × 2 accumulation = 16
# - 相比单卡训练，速度提升约3-3.5倍
