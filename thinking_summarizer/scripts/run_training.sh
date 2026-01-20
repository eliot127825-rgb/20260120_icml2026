#!/bin/bash

# Thinking Summarizer 训练启动脚本

export CUDA_VISIBLE_DEVICES=0

python train_summarizer.py \
  --model-path ${PROJECT_ROOT}/Qwen2.5-3B-Instruct \
  --train-file ./data/training_dataset/train.json \
  --val-file ./data/training_dataset/val.json \
  --output-dir ./outputs/thinking_summarizer \
  --max-length 2048 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --num-epochs 3 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --learning-rate 3e-4 \
  --warmup-ratio 0.1 \
  --save-steps 100 \
  --logging-steps 10
