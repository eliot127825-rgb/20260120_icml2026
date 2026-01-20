#!/usr/bin/env python3
"""合并4个worker的summary结果"""

import json
from pathlib import Path

base_dir = Path('${PROJECT_ROOT}/thinking_summarizer/data/thinking_summaries_6772')

all_summaries = []
for i in range(4):
    file = base_dir / f'workers/worker_{i}/summaries_all.json'
    if file.exists():
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_summaries.extend(data)
            print(f'Worker {i}: {len(data)} 条')

output_file = base_dir / 'summaries_all.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_summaries, f, indent=2, ensure_ascii=False)

print(f'\n✓ 合并完成，共 {len(all_summaries)} 条')
print(f'输出文件: {output_file}')

# 统计信息
print(f'\n📊 数据来源统计:')
sources = {}
for item in all_summaries:
    src = item.get('metadata', {}).get('source', 'unknown')
    sources[src] = sources.get(src, 0) + 1
for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
    print(f'   {src}: {cnt}')
