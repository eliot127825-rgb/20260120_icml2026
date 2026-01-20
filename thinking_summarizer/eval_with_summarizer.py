#!/usr/bin/env python3
"""
集成evaluation脚本：HumanOmniV2模型生成thinking + Summarizer压缩
"""
import os
import sys
import json
import re
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

# 添加eval路径
sys.path.insert(0, '${PROJECT_ROOT}/src/eval')
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from qwen_omni_utils import process_mm_info

def extract_think(output_str):
    """提取<think>标签中的内容"""
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    """提取<answer>标签中的内容"""
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

class SummarizerModel:
    """Summarizer模型封装"""
    def __init__(self, base_model_path, lora_path):
        print("加载Summarizer模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        print("✓ Summarizer加载完成")
    
    def summarize(self, thinking_text):
        """对thinking进行总结"""
        instruction = """Analyze the following reasoning process and extract structured information.

Output format:
## Key Points Analysis
1. [First key point]
2. [Second key point]
3. [Third key point]

## Video Focus Objects
- People: [List important people with appearance and emotional features]
- Objects: [List important objects]
- Scenes: [List important scene elements]

## Emotional Indicators (if applicable)
- [List specific visual cues indicating emotions]

## SAM3 Segmentation Instructions
Please segment the following in the video: [object1], [object2], [object3]"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{instruction}\n\nThinking Process:\n{thinking_text}"}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
        
        summary = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return summary

def load_test_samples(data_path, video_dir=None, num_samples=5):
    """加载测试样本，支持多种数据格式"""
    with open(data_path) as f:
        data = json.load(f)
    
    # 取前num_samples个样本
    samples = []
    for item in data[:num_samples]:
        # 获取视频路径
        video = item.get('video_path') or item.get('video', '')
        if video_dir and not os.path.isabs(video):
            video = os.path.join(video_dir, video)
        
        # 构建问题文本（支持多种格式）
        if 'question' in item:
            question = item['question']
        else:
            # 使用 problem + options 格式
            question = item.get('problem', '')
            if 'options' in item:
                options_text = '\n'.join(item['options'])
                question = f"{question} Options:\n{options_text}\n Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        
        samples.append({
            'qid': item.get('qid', ''),
            'video_path': video,
            'question': question,
            'problem_type': item.get('problem_type', ''),
            'solution': item.get('solution', '')
        })
    
    return samples

def run_evaluation(args):
    """运行完整的evaluation流程"""
    
    # 1. 加载HumanOmniV2模型
    print("="*80)
    print("Step 1: 加载HumanOmniV2 Thinker模型")
    print("="*80)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.humanomniv2_model)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.humanomniv2_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ HumanOmniV2模型加载完成\n")
    
    # 2. 加载Summarizer模型
    print("="*80)
    print("Step 2: 加载Thinking Summarizer模型")
    print("="*80)
    summarizer = SummarizerModel(args.base_model, args.lora_model)
    print()
    
    # 3. 加载测试样本
    print("="*80)
    print(f"Step 3: 加载测试样本 (前{args.num_samples}个)")
    print("="*80)
    test_samples = load_test_samples(args.data_path, args.video_dir, args.num_samples)
    print(f"✓ 加载了 {len(test_samples)} 个测试样本\n")
    
    # 4. 运行pipeline
    print("="*80)
    print("Step 4: 运行完整Pipeline")
    print("="*80)
    
    results = []
    
    for idx, sample in enumerate(tqdm(test_samples, desc="处理样本")):
        print(f"\n{'='*80}")
        print(f"样本 {idx+1}/{len(test_samples)}")
        print(f"{'='*80}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Video: {sample['video_path']}")
        
        # 4.1 构建输入
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample['video_path']},
                    {"type": "text", "text": sample['question']}
                ]
            }
        ]
        
        # process_mm_info返回(audios, images, videos)
        audios, images, videos = process_mm_info(message, use_audio_in_video=False)
        
        # 应用chat template
        text = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 构建模型输入
        model_inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        
        # 4.2 生成thinking
        print("\n生成thinking...")
        with torch.inference_mode():
            text_ids = model.generate(
                **model_inputs.to(model.device).to(model.dtype),
                use_audio_in_video=False,
                max_new_tokens=2048
            )
        
        full_output = processor.decode(text_ids[0][model_inputs.input_ids.size(1):], skip_special_tokens=True)
        thinking = extract_think(full_output)
        answer = extract_answer(full_output)
        
        print(f"✓ Thinking生成完成 (长度: {len(thinking)} 字符)")
        print(f"✓ Answer: {answer}")
        
        # 打印完整thinking
        print("\n" + "="*80)
        print("【原模型完整输出】")
        print("="*80)
        print(f"\n>>> Full Output (长度: {len(full_output)} 字符):\n")
        print(full_output)
        print("\n" + "-"*80)
        print(f">>> Thinking (长度: {len(thinking)} 字符):\n")
        print(thinking)
        print("\n" + "-"*80)
        print(f">>> Answer:\n")
        print(answer)
        print("="*80)
        
        # 4.3 总结thinking
        print("\n总结thinking...")
        summary = summarizer.summarize(thinking)
        print(f"✓ Summary生成完成 (长度: {len(summary)} 字符)")
        
        # 打印完整summary
        print("\n" + "="*80)
        print("【Summarizer模型完整输出】")
        print("="*80)
        print(summary)
        print("="*80)
        
        # 保存结果
        result = {
            'qid': sample['qid'],
            'question': sample['question'],
            'video_path': sample['video_path'],
            'ground_truth': sample['solution'],
            'full_output': full_output,
            'thinking': thinking,
            'answer': answer,
            'summary': summary
        }
        results.append(result)
    
    # 5. 保存结果
    output_file = args.output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("Pipeline完成！")
    print(f"{'='*80}")
    print(f"✓ 结果已保存到: {output_file}")
    print(f"✓ 处理了 {len(results)} 个样本")
    
    # 统计信息
    print(f"\n{'='*80}")
    print("统计信息:")
    print(f"{'='*80}")
    avg_thinking_len = sum(len(r['thinking']) for r in results) / len(results)
    avg_summary_len = sum(len(r['summary']) for r in results) / len(results)
    compression_ratio = (1 - avg_summary_len / avg_thinking_len) * 100 if avg_thinking_len > 0 else 0
    
    print(f"平均Thinking长度: {avg_thinking_len:.0f} 字符")
    print(f"平均Summary长度: {avg_summary_len:.0f} 字符")
    print(f"压缩率: {compression_ratio:.1f}%")
    print(f"\n详细结果已保存到JSON文件，可查看完整内容。")

def main():
    parser = argparse.ArgumentParser(description="集成evaluation: HumanOmniV2 + Summarizer")
    
    # HumanOmniV2模型
    parser.add_argument(
        '--humanomniv2-model',
        type=str,
        default='os.environ.get('MODEL_CHECKPOINT_DIR', './checkpoints')/stage5_outcome_reward/checkpoint-1083',
        help='HumanOmniV2模型路径'
    )
    
    # Summarizer模型
    parser.add_argument(
        '--base-model',
        type=str,
        default='${PROJECT_ROOT}/Qwen2.5-3B-Instruct',
        help='Summarizer基座模型路径'
    )
    parser.add_argument(
        '--lora-model',
        type=str,
        default='${PROJECT_ROOT}/thinking_summarizer/outputs/thinking_summarizer_6770/final_model',
        help='Summarizer LoRA权重路径'
    )
    
    # 数据
    parser.add_argument(
        '--data-path',
        type=str,
        default='${PROJECT_ROOT}/thinking_summarizer/data/my_demo/demo_test_data.json',
        help='测试数据路径'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='${PROJECT_ROOT}/thinking_summarizer/data/my_demo',
        help='视频文件目录'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='测试样本数量'
    )
    
    # 输出
    parser.add_argument(
        '--output-file',
        type=str,
        default='./outputs/eval_with_summarizer_results.json',
        help='输出结果文件'
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    run_evaluation(args)

if __name__ == '__main__':
    main()
