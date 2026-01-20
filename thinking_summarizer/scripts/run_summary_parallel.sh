#!/bin/bash

set -e

INPUT_FILE="./data/thinking_outputs_all_6772/thinking_data_all.json"
OUTPUT_BASE="./data/thinking_summaries_6772"
API_KEY="sk-27ed7c677b304d729eb953f698e828f9"
MODEL="qwen-max"
TOTAL_SAMPLES=6772
NUM_WORKERS=4

echo "================================================================================"
echo "4进程并行Summary生成脚本"
echo "================================================================================"
echo "输入文件: $INPUT_FILE"
echo "输出路径: $OUTPUT_BASE"
echo "总样本数: $TOTAL_SAMPLES"
echo "并行数: $NUM_WORKERS"
echo "================================================================================"

# 计算每个worker处理的样本数
SAMPLES_PER_WORKER=$((TOTAL_SAMPLES / NUM_WORKERS))
echo "每个进程处理: ~$SAMPLES_PER_WORKER 条"

# 创建输出目录
mkdir -p "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE/logs"
mkdir -p "$OUTPUT_BASE/workers"

echo ""
echo "分割输入数据..."
python3 << EOF
import json

with open('$INPUT_FILE', 'r') as f:
    data = json.load(f)

total = len(data)
chunk_size = (total + $NUM_WORKERS - 1) // $NUM_WORKERS

for i in range($NUM_WORKERS):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, total)
    chunk = data[start:end]
    
    output_file = f'$OUTPUT_BASE/workers/input_worker_{i}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)
    
    print(f"Worker {i}: {len(chunk)} 条 (索引 {start}-{end-1})")

print(f"✓ 数据分割完成，共 {total} 条")
EOF

echo ""
echo "启动 $NUM_WORKERS 个并行进程..."

for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    INPUT_WORKER="$OUTPUT_BASE/workers/input_worker_${WORKER_ID}.json"
    OUTPUT_WORKER="$OUTPUT_BASE/workers/worker_${WORKER_ID}"
    LOG_FILE="$OUTPUT_BASE/logs/worker_${WORKER_ID}.log"
    
    echo "  启动Worker $WORKER_ID..."
    
    nohup python3 scripts/call_api_summarize.py \
        --input-path "$INPUT_WORKER" \
        --output-path "$OUTPUT_WORKER" \
        --api-type qwen \
        --api-key "$API_KEY" \
        --model "$MODEL" \
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
echo "  查看所有日志: tail -f $OUTPUT_BASE/logs/worker_*.log"
echo "  查看单个进程: tail -f $OUTPUT_BASE/logs/worker_0.log"
echo "  检查进程: ps aux | grep call_api_summarize.py"
echo ""
echo "等待所有进程完成后，运行合并命令:"
echo "  python3 << 'MERGE_EOF'"
echo "  import json"
echo "  from pathlib import Path"
echo "  "
echo "  all_summaries = []"
echo "  for i in range($NUM_WORKERS):"
echo "      file = Path('$OUTPUT_BASE/workers/worker_{i}/summaries_all.json'.format(i=i))"
echo "      if file.exists():"
echo "          with open(file, 'r') as f:"
echo "              data = json.load(f)"
echo "              all_summaries.extend(data)"
echo "              print(f'Worker {i}: {len(data)} 条')"
echo "  "
echo "  output_file = '$OUTPUT_BASE/summaries_all.json'"
echo "  with open(output_file, 'w', encoding='utf-8') as f:"
echo "      json.dump(all_summaries, f, indent=2, ensure_ascii=False)"
echo "  "
echo "  print(f'✓ 合并完成，共 {len(all_summaries)} 条')"
echo "  print(f'输出文件: {output_file}')"
echo "  MERGE_EOF"
echo ""
echo "================================================================================"
