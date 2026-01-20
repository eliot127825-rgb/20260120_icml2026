#!/usr/bin/env python3
"""
合并多个shard的thinking输出
"""

import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Merge thinking output shards')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing shard files')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output merged file path')
    parser.add_argument('--num-shards', type=int, required=True,
                        help='Number of shards to merge')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    print("=" * 80)
    print("合并Thinking输出文件")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"分片数量: {args.num_shards}")
    print("=" * 80)
    
    # 读取所有shard
    all_results = []
    for i in range(args.num_shards):
        shard_file = input_dir / f'thinking_data_shard_{i}.json'
        
        if not shard_file.exists():
            print(f"⚠️  Shard {i} 不存在: {shard_file}")
            continue
        
        print(f"\n读取 Shard {i}...")
        with open(shard_file, 'r', encoding='utf-8') as f:
            shard_data = json.load(f)
        
        print(f"  ✓ 加载 {len(shard_data)} 个样本")
        all_results.extend(shard_data)
    
    # 去重（基于video_id + question，因为同一视频可能有多个不同问题）
    print(f"\n合并前: {len(all_results)} 个样本")
    seen_keys = set()
    unique_results = []
    for item in all_results:
        # 使用video_id + question作为唯一标识
        # 如果有qid则优先使用qid
        qid = item.get('metadata', {}).get('qid', '')
        if qid:
            key = qid
        else:
            key = (item['video_id'], item.get('question', ''))
        
        if key not in seen_keys:
            seen_keys.add(key)
            unique_results.append(item)
    
    print(f"去重后: {len(unique_results)} 个样本 (移除了 {len(all_results) - len(unique_results)} 个真正的重复)")
    
    # 保存
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"✅ 完成！共合并 {len(unique_results)} 个thinking样本")
    print(f"输出文件: {output_path}")
    print("=" * 80)
    
    # 统计
    thinking_lengths = [len(r['thinking']) for r in unique_results if r['thinking']]
    if thinking_lengths:
        print(f"\nThinking统计:")
        print(f"  平均长度: {sum(thinking_lengths) // len(thinking_lengths)} 字符")
        print(f"  最短: {min(thinking_lengths)} 字符")
        print(f"  最长: {max(thinking_lengths)} 字符")
    
    # 数据来源统计
    source_counts = {}
    for item in unique_results:
        source = item['metadata'].get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    if source_counts:
        print("\n数据来源统计:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count} 样本")

if __name__ == '__main__':
    main()
