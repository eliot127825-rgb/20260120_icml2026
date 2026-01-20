#!/bin/bash

set -e

cd ${PROJECT_ROOT}/thinking_summarizer

echo "================================================================================"
echo "LoRA模型训练 - Thinking Summarizer"
echo "================================================================================"

# 使用GPU 4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 训练配置
MODEL_PATH="${PROJECT_ROOT}/Qwen2.5-3B-Instruct"
TRAIN_FILE="./data/training_dataset_6770/train.json"
VAL_FILE="./data/training_dataset_6770/val.json"
OUTPUT_DIR="./outputs/thinking_summarizer_6770"

echo "基座模型: $MODEL_PATH"
echo "训练数据: $TRAIN_FILE (6093条)"
echo "验证数据: $VAL_FILE (677条)"
echo "输出目录: $OUTPUT_DIR"
echo "GPU: 4,5,6,7"
echo "================================================================================"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p ./logs

# 多GPU训练 (使用torchrun)
torchrun --nproc_per_node=4 --master_port=29501 \
    scripts/train_summarizer.py \
    --model-path $MODEL_PATH \
    --train-file $TRAIN_FILE \
    --val-file $VAL_FILE \
    --output-dir $OUTPUT_DIR \
    --num-epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.1 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --max-length 2048 \
    --logging-steps 10 \
    --save-steps 200

echo ""
echo "================================================================================"
echo "✓ 训练完成！"
echo "模型保存位置: $OUTPUT_DIR/final_model"
echo "================================================================================"
