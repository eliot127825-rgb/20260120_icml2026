#!/usr/bin/env python3
"""
批量生成HumanOmniV2模型的thinking输出
用于构建thinking总结模型的训练数据
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import re
import av

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "spatio-temporal-reasoner" / "src"))

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

def check_if_video_has_audio(video_path: str) -> bool:
    """检查视频是否包含音频流"""
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False

def extract_thinking(text: str) -> str:
    """从模型输出中提取thinking部分"""
    think_pattern = r'<think>\s*(.*?)\s*</think>'
    matches = re.findall(think_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return ""

def extract_answer(text: str) -> str:
    """从模型输出中提取answer部分"""
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = re.findall(answer_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text

def construct_message(video_path: str, question: str, problem_type: str = 'multiple choice', data_type: str = 'video') -> List[Dict]:
    """构造模型输入消息格式（与evaluation完全一致）"""
    video_audio_available = check_if_video_has_audio(video_path)
    
    # 使用evaluation时的原始system prompt（保持训练一致性）
    system_prompt = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question. 

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."""
    
    # 添加问题类型的instruction（与evaluation完全一致）
    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    }
    
    question_with_instruction = question + TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE['free-form'])
    
    if video_audio_available:
        user_content = [
            {"type": data_type, data_type: video_path},
            {"type": "audio", "audio": video_path},
            {"type": "text", "text": f"Here is a {data_type}, with the audio from the video.\n{question_with_instruction}"}
        ]
    else:
        user_content = [
            {"type": data_type, data_type: video_path},
            {"type": "text", "text": question_with_instruction}
        ]
    
    message = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content}
    ]
    
    return message

def load_video_dataset(data_path: str) -> List[Dict]:
    """加载视频数据集并提取原始问题"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 适配不同数据格式
    if isinstance(data, dict):
        # WorldSense格式
        videos = []
        for video_id, video_data in data.items():
            # WorldSense可能有多个task，每个task是一个问题
            for key, value in video_data.items():
                if key.startswith('task'):
                    videos.append({
                        'video_id': video_id,
                        'video_path': f"${DATA_ROOT}/HoEvalDate/Videos/{video_id}/{video_id}_video.mp4",
                        'question': value.get('question', ''),
                        'task_id': key,
                        'metadata': value
                    })
        return videos
    elif isinstance(data, list):
        # 检查是否已经是统一格式（由sample_datasets.py生成）
        if data and 'question' in data[0] and 'video_path' in data[0]:
            # 已处理的统一格式，直接返回
            return data
        
        # Daily-Omni原始格式
        videos = []
        for item in data:
            video_id = item.get('video_id', '')
            # 提取原始问题和选项
            question = item.get('Question', '')
            choices = item.get('Choice', [])
            
            # 构造完整问题（问题 + 选项，与evaluation一致）
            if choices:
                full_question = question + " Options:\n"
                for choice in choices:
                    full_question += choice + "\n"
            else:
                full_question = question
            
            videos.append({
                'video_id': video_id,
                'video_path': f"${DATA_ROOT}/HoEvalDate/Videos/{video_id}/{video_id}_video.mp4",
                'question': full_question,
                'problem_type': 'multiple choice',  # Daily-Omni都是选择题
                'data_type': 'video',  # 与eval保持一致
                'metadata': item
            })
        return videos
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

def main():
    parser = argparse.ArgumentParser(description='Generate thinking outputs for training data')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to HumanOmniV2 model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to video dataset JSON')
    parser.add_argument('--output-path', type=str, default='./thinking_outputs',
                        help='Output directory for thinking data')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--extract-thinking-only', action='store_true',
                        help='Only extract thinking part from output')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='Shard ID for multi-GPU processing (0-based)')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Total number of shards for multi-GPU processing')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Thinking数据生成配置")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output_path}")
    print(f"使用数据集原始问题: 是")
    if args.num_shards > 1:
        print(f"多GPU模式: Shard {args.shard_id + 1}/{args.num_shards}")
    print("=" * 80)
    
    # 加载数据集
    print("\n加载视频数据集...")
    videos = load_video_dataset(args.data_path)
    
    # 数据分片（多GPU支持）
    if args.num_shards > 1:
        total_samples = len(videos)
        shard_size = (total_samples + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_samples)
        videos = videos[start_idx:end_idx]
        print(f"数据分片: 处理样本 {start_idx}-{end_idx-1} (共{len(videos)}个)")
    
    if args.max_samples:
        videos = videos[:args.max_samples]
    
    print(f"本进程处理 {len(videos)} 个视频")
    
    # 加载模型
    print("\n加载HumanOmniV2模型...")
    print(f"模型路径: {args.model_path}")
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model.eval()
    
    print(f"✓ 模型加载完成")
    print(f"  设备: {model.device}")
    print(f"  数据类型: {model.dtype}")
    
    # 批量生成thinking
    print("\n开始生成thinking输出...")
    results = []
    
    for i, video_info in enumerate(tqdm(videos)):
        try:
            # 检查视频文件是否存在
            if not os.path.exists(video_info['video_path']):
                print(f"\n⚠️  视频文件不存在: {video_info['video_path']}")
                continue
            
            # 使用数据集的原始问题构造输入消息
            question = video_info.get('question', '')
            problem_type = video_info.get('problem_type', 'multiple choice')
            data_type = video_info.get('data_type', 'video')
            
            if not question:
                print(f"\n⚠️  {video_info['video_id']} 没有问题，跳过")
                continue
            
            message = construct_message(video_info['video_path'], question, problem_type, data_type)
            
            # 处理多模态信息
            audios, images, videos = process_mm_info(message, use_audio_in_video=False)
            
            # 应用聊天模板
            text = processor.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 准备模型输入
            model_inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False
            )
            
            model_inputs = model_inputs.to(model.device).to(model.dtype)
            
            # 生成输出
            with torch.inference_mode():
                text_ids = model.generate(
                    **model_inputs,
                    use_audio_in_video=False,
                    max_new_tokens=2048
                )
            
            # 解码输出
            input_length = model_inputs.input_ids.size(1)
            output = processor.decode(
                text_ids[0][input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # 提取thinking
            thinking = extract_thinking(output)
            answer = extract_answer(output)
            
            result = {
                'video_id': video_info['video_id'],
                'video_path': video_info['video_path'],
                'question': question,  # 保存原始问题
                'problem_type': problem_type,
                'thinking': thinking,
                'answer': answer if not args.extract_thinking_only else None,
                'full_output': output if not args.extract_thinking_only else None,
                'metadata': video_info.get('metadata', {})
            }
            
            results.append(result)
            
            # 定期保存
            if (i + 1) % 100 == 0:
                output_file = output_dir / f'thinking_data_batch_{i+1}.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results[-100:], f, ensure_ascii=False, indent=2)
                print(f"\n已保存批次数据: {output_file}")
        
        except Exception as e:
            print(f"\n处理 {video_info['video_id']} 时出错: {str(e)}")
            continue
    
    # 保存最终结果
    if args.num_shards > 1:
        output_file = output_dir / f'thinking_data_shard_{args.shard_id}.json'
    else:
        output_file = output_dir / 'thinking_data_all.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"✅ 完成！共生成 {len(results)} 个thinking样本")
    print(f"输出文件: {output_file}")
    print("=" * 80)
    
    # 统计信息
    thinking_lengths = [len(r['thinking']) for r in results if r['thinking']]
    if thinking_lengths:
        print(f"\nThinking统计:")
        print(f"  平均长度: {sum(thinking_lengths) / len(thinking_lengths):.0f} 字符")
        print(f"  最短: {min(thinking_lengths)} 字符")
        print(f"  最长: {max(thinking_lengths)} 字符")

if __name__ == '__main__':
    main()
