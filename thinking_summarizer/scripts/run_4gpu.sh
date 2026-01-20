#!/bin/bash
# 4卡并行生成thinking outputs

# 配置
MODEL_PATH="../spatio-temporal-reasoner/outputs/stage4_debug_no_audio_v2/checkpoint-380"
DATA_PATH="./data/sampled_mixed_1000.json"
OUTPUT_PATH="./data/thinking_outputs_mixed_1000_4gpu"
NUM_SHARDS=4

echo "========================================"
echo "4卡并行生成Thinking"
echo "========================================"
echo "模型: $MODEL_PATH"
echo "数据: $DATA_PATH"
echo "输出: $OUTPUT_PATH"
echo "分片数: $NUM_SHARDS"
echo "========================================"

# 创建输出目录
mkdir -p "$OUTPUT_PATH"

# 启动4个进程，每个使用不同的GPU
for i in {0..3}; do
    echo "启动 GPU $i (Shard $i)..."
    CUDA_VISIBLE_DEVICES=$i python generate_thinking.py \
        --model-path "$MODEL_PATH" \
        --data-path "$DATA_PATH" \
        --output-path "$OUTPUT_PATH" \
        --shard-id $i \
        --num-shards $NUM_SHARDS \
        --extract-thinking-only \
        > "$OUTPUT_PATH/shard_${i}.log" 2>&1 &
    
    # 记录进程ID
    echo $! >> "$OUTPUT_PATH/pids.txt"
    
    # 稍微延迟避免同时加载模型
    sleep 5
done

echo ""
echo "所有进程已启动！"
echo "进程ID已保存到: $OUTPUT_PATH/pids.txt"
echo ""
echo "查看进度:"
echo "  tail -f $OUTPUT_PATH/shard_0.log"
echo "  tail -f $OUTPUT_PATH/shard_1.log"
echo "  tail -f $OUTPUT_PATH/shard_2.log"
echo "  tail -f $OUTPUT_PATH/shard_3.log"
echo ""
echo "监控所有进程:"
echo "  watch -n 5 'ps aux | grep generate_thinking.py'"
echo ""
echo "等待所有进程完成:"
echo "  wait \$(cat $OUTPUT_PATH/pids.txt)"
