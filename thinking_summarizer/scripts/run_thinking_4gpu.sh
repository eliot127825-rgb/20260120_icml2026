#!/bin/bash

set -e

MODEL_PATH="${PROJECT_ROOT}/outputs/stage4_debug_no_audio_v2/checkpoint-380"
DATA_PATH="./data/sampled_mixed_all.json"
OUTPUT_PATH="./data/thinking_outputs_all_2gpu"
MAX_SAMPLES=999999  # 设置为足够大的值，确保全量生成
NUM_SHARDS=2

echo "================================================================================"
echo "4GPU并行Thinking生成脚本"
echo "================================================================================"
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "总样本数: $MAX_SAMPLES"
echo "GPU数量: $NUM_SHARDS"
echo "================================================================================"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH/logs"

echo "清理旧的日志文件..."
rm -f "$OUTPUT_PATH"/logs/shard_*.log

echo ""
echo "启动2个GPU进程（GPU 4,5）..."

for SHARD_ID in 0 1; do
    GPU_ID=$((SHARD_ID + 4))  # GPU 4, 5
    LOG_FILE="$OUTPUT_PATH/logs/shard_${SHARD_ID}.log"
    
    echo "  启动GPU $GPU_ID (Shard $SHARD_ID)..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python3 scripts/generate_thinking.py \
        --model-path "$MODEL_PATH" \
        --data-path "$DATA_PATH" \
        --output-path "$OUTPUT_PATH" \
        --shard-id $SHARD_ID \
        --num-shards $NUM_SHARDS \
        --max-samples $MAX_SAMPLES \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "    PID: $PID, 日志: $LOG_FILE"
    
    sleep 2
done

echo ""
echo "================================================================================"
echo "✓ 所有进程已启动"
echo "================================================================================"
echo ""
echo "监控命令:"
echo "  查看所有日志: tail -f $OUTPUT_PATH/logs/shard_*.log"
echo "  查看单个GPU: tail -f $OUTPUT_PATH/logs/shard_0.log"
echo "  查看进度: watch -n 30 'ls -lh $OUTPUT_PATH/thinking_data_shard_*.json'"
echo "  检查进程: ps aux | grep generate_thinking.py"
echo ""
echo "等待所有进程完成后，运行合并命令:"
echo "  python3 scripts/merge_shards.py \\"
echo "    --input-dir $OUTPUT_PATH \\"
echo "    --output-file $OUTPUT_PATH/thinking_data_all.json"
echo ""
echo "================================================================================"
