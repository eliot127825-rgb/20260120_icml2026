#!/usr/bin/env python3
"""
将API总结数据转换为训练数据集
格式：instruction-input-output，适合微调Qwen模型
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

INSTRUCTION_TEMPLATE = """You are an expert in analyzing reasoning processes and extracting structured information. Given a detailed thinking process about video content analysis, extract key points and generate SAM3 segmentation instructions.

Please output in the following format:

## Key Points Analysis
1. [First key point]
2. [Second key point]
3. [Third key point]

## Video Focus Objects
- People: [List all important people with their appearance features]
- Objects: [List all important objects]
- Scenes: [List important scene elements]

## Emotional Indicators (if applicable)
- [List specific visual cues that indicate emotions, such as facial expressions, body language, tone, gestures, etc.]

## SAM3 Segmentation Instructions
Please segment the following in the video: [object1], [object2], [object3]

Note: List exactly 3 objects separated by commas on a single line.

Requirements:
1. Key points should be concise and clear, each within 30 words
2. Extract only content relevant to visual segmentation
3. For people analysis, include:
   - Physical appearance (e.g., "woman in red dress")
   - Facial expressions and emotions (e.g., "smiling man", "frustrated woman with furrowed brows")
   - Body language indicators (e.g., "person with crossed arms", "nodding")
4. Object descriptions should include location information (e.g., "vase on the table")
5. If the video involves people or character scenes, prioritize listing visual details that help infer emotions (facial expressions, posture, gestures, tone indicators)
6. **IMPORTANT**: SAM3 Segmentation Instructions must list ONLY the top 3 most important objects/people to segment. Choose the most critical elements from the analysis.
7. Output must strictly follow the above format"""

def format_summary_as_output(summary_text: str) -> str:
    """将API返回的总结文本格式化为训练输出"""
    # 直接使用API返回的完整总结文本
    return summary_text.strip()

def build_training_sample(item: Dict) -> Dict:
    """构建单个训练样本"""
    return {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": item["original_thinking"],
        "output": item["summary_text"],
        "metadata": {
            "video_id": item["video_id"],
            "source": item.get("metadata", {}).get("source", ""),
            "type": item.get("metadata", {}).get("Type", ""),
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Build training dataset from API summaries')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input summaries_all.json file')
    parser.add_argument('--output-dir', type=str, default='./data/training_dataset',
                        help='Output directory for training data')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Ratio of training samples (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("构建训练数据集")
    print("=" * 80)
    print(f"输入文件: {args.input_file}")
    print(f"输出路径: {args.output_dir}")
    print(f"训练集比例: {args.train_ratio}")
    print("=" * 80)
    
    # 加载API总结数据
    print("\n加载API总结数据...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    print(f"共加载 {len(summaries)} 个总结样本")
    
    # 构建训练样本
    print("\n构建训练样本...")
    training_samples = []
    
    for item in summaries:
        try:
            sample = build_training_sample(item)
            training_samples.append(sample)
        except Exception as e:
            print(f"⚠️  处理 {item.get('video_id', 'unknown')} 时出错: {str(e)}")
            continue
    
    print(f"成功构建 {len(training_samples)} 个训练样本")
    
    # 随机打乱
    random.shuffle(training_samples)
    
    # 划分训练集和验证集
    split_idx = int(len(training_samples) * args.train_ratio)
    train_samples = training_samples[:split_idx]
    val_samples = training_samples[split_idx:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本")
    
    # 保存训练集
    train_file = output_dir / 'train.json'
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 训练集已保存: {train_file}")
    
    # 保存验证集
    val_file = output_dir / 'val.json'
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"✅ 验证集已保存: {val_file}")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("数据集统计")
    print("=" * 80)
    
    # 统计数据来源
    sources = {}
    for sample in training_samples:
        source = sample['metadata']['source']
        sources[source] = sources.get(source, 0) + 1
    
    print("\n数据来源分布:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} ({count/len(training_samples)*100:.1f}%)")
    
    # 统计输入输出长度
    input_lengths = [len(s['input']) for s in training_samples]
    output_lengths = [len(s['output']) for s in training_samples]
    
    print(f"\n输入长度统计:")
    print(f"  平均: {sum(input_lengths)/len(input_lengths):.0f} 字符")
    print(f"  最短: {min(input_lengths)} 字符")
    print(f"  最长: {max(input_lengths)} 字符")
    
    print(f"\n输出长度统计:")
    print(f"  平均: {sum(output_lengths)/len(output_lengths):.0f} 字符")
    print(f"  最短: {min(output_lengths)} 字符")
    print(f"  最长: {max(output_lengths)} 字符")
    
    print("\n" + "=" * 80)
    print("✅ 数据集构建完成！")
    print("=" * 80)
    
    # 保存一个示例
    example_file = output_dir / 'example.json'
    with open(example_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples[0], f, ensure_ascii=False, indent=2)
    print(f"\n示例样本已保存: {example_file}")

if __name__ == '__main__':
    main()
