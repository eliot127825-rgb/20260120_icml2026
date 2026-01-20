#!/bin/bash

set -e

cd ${PROJECT_ROOT}/thinking_summarizer

echo "================================================================================"
echo "测试API调用 - 10条样本"
echo "================================================================================"

# 1. 创建测试数据（前10条）
echo "1. 创建测试数据..."
python3 << 'EOF'
import json

with open('./data/thinking_outputs_all_6772/thinking_data_all.json', 'r') as f:
    data = json.load(f)

test_data = data[:10]

with open('./data/test_10_samples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f'✓ 已创建测试数据: {len(test_data)} 条')
for i, item in enumerate(test_data):
    print(f'  {i+1}. {item["video_id"]} - {item["metadata"].get("source", "unknown")}')
EOF

# 2. 运行API测试
echo ""
echo "2. 调用API生成summary..."
echo "================================================================================"

python3 scripts/call_api_summarize.py \
    --input-path ./data/test_10_samples.json \
    --output-path ./data/test_summaries_10 \
    --api-type qwen \
    --api-key "sk-27ed7c677b304d729eb953f698e828f9" \
    --model qwen-max

echo ""
echo "================================================================================"
echo "✓ 测试完成！"
echo "================================================================================"
echo "结果文件: ./data/test_summaries_10/summaries_all.json"
echo ""
echo "查看结果:"
echo "  cat ./data/test_summaries_10/summaries_all.json | head -100"
