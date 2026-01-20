#!/usr/bin/env python3
"""
完整Pipeline: HumanOmniV2 -> Summarizer -> SAM3
使用SAM3官方API进行视频分割
"""
import os
import sys
import json
import argparse
import re
from typing import List, Dict
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import numpy as np
from PIL import Image

# 添加SAM3路径
sys.path.insert(0, os.environ.get('SAM3_CODE_PATH', './sam3/sam3_code'))

# 添加eval路径
sys.path.insert(0, '${PROJECT_ROOT}/src/eval')
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from qwen_omni_utils import process_mm_info

# 导入SAM3
from sam3.model_builder import build_sam3_video_predictor

def simplify_prompt_for_sam3(prompt: str) -> str:
    """
    将复杂的prompt简化为SAM3友好的简单类别名称
    
    Args:
        prompt: 原始复杂prompt，如"President John F. Kennedy"
    
    Returns:
        简化后的prompt，如"person"
    """
    prompt_lower = prompt.lower()
    
    # 人物相关
    if any(keyword in prompt_lower for keyword in [
        'president', 'speaker', 'presenter', 'doctor', 'dr.', 'man', 'woman',
        'person', 'people', 'kennedy', 'host', 'announcer', 'reporter'
    ]):
        return "person"
    
    # 屏幕/显示设备
    if any(keyword in prompt_lower for keyword in [
        'television', 'tv', 'monitor', 'display', 'screen', 'broadcast'
    ]):
        return "screen"
    
    # 产品/物体
    if any(keyword in prompt_lower for keyword in [
        'product', 'bottle', 'container', 'device', 'tool', 'equipment'
    ]):
        return "object"
    
    # 文字/文本
    if any(keyword in prompt_lower for keyword in [
        'text', 'overlay', 'subtitle', 'caption', 'label', 'title'
    ]):
        return "text"
    
    # 家具
    if any(keyword in prompt_lower for keyword in [
        'table', 'chair', 'desk', 'shelf', 'bookshelf', 'furniture'
    ]):
        return "furniture"
    
    # 如果没有匹配，返回原prompt（去除修饰词后的核心名词）
    # 移除形容词和所有格，保留核心名词
    simplified = re.sub(r'\b(simulated|vintage|old|new|large|small|big|red|blue)\b', '', prompt_lower).strip()
    simplified = re.sub(r"'s\b", '', simplified).strip()
    simplified = re.sub(r'\s+', ' ', simplified)  # 合并多余空格
    
    return simplified if simplified else prompt

def extract_context(output_str):
    """提取<context>标签中的内容"""
    pattern = r'<context>\s*(.*?)\s*</context>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

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

def parse_sam3_instructions(summary):
    """从summary中提取SAM3分割指令"""
    pattern = r'## SAM3 Segmentation Instructions\s*\n\s*Please segment the following in the video:\s*(.+?)(?:\n\n|\Z)'
    match = re.search(pattern, summary, re.DOTALL | re.IGNORECASE)
    if match:
        objects_str = match.group(1).strip()
        # 分割对象列表
        objects = [obj.strip() for obj in objects_str.split(',')]
        return objects
    return []

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

class SAM3Segmenter:
    """SAM3视频分割封装"""
    def __init__(self, checkpoint_path=None, gpu_id=1):
        print(f"加载SAM3视频预测器 (GPU {gpu_id})...")
        self.gpu_id = gpu_id
        
        # 临时切换到指定 GPU 加载模型
        import torch
        with torch.cuda.device(gpu_id):
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"  使用本地checkpoint: {checkpoint_path}")
                self.predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path)
            else:
                print("  使用HuggingFace自动下载")
                self.predictor = build_sam3_video_predictor()
            print("✓ SAM3加载完成")
    
    def segment_video(self, video_path, text_prompts, max_frames=30):
        """
        对视频执行SAM3分割
        
        Args:
            video_path: 视频文件路径
            text_prompts: 文本提示列表，如 ["person", "car"]
            max_frames: 最大处理帧数（默认30帧，约1-2秒视频）
        
        Returns:
            分割结果字典，每个prompt的分割结果
        """
        print(f"\n处理视频: {video_path}")
        
        results_per_prompt = {}
        
        for prompt_idx, text_prompt in enumerate(text_prompts):
            print(f"\n  [{prompt_idx+1}/{len(text_prompts)}] 分割对象: '{text_prompt}'")
            
            try:
                # 1. 启动session
                response = self.predictor.handle_request(
                    request=dict(
                        type="start_session",
                        resource_path=video_path,
                    )
                )
                session_id = response["session_id"]
                
                # 2. 添加文本prompt在第0帧
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=text_prompt,
                    )
                )
                print(f"    [调试] add_prompt响应: frame_index={response.get('frame_index')}, outputs keys={list(response.get('outputs', {}).keys())}")
                if 'outputs' in response and 'out_obj_ids' in response['outputs']:
                    print(f"    [调试] add_prompt返回的out_obj_ids: {response['outputs']['out_obj_ids']}")
                
                # 3. 传播分割到整个视频（这是关键步骤！）
                all_outputs = {}
                frame_count = 0
                for result_dict in self.predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        propagation_direction="forward",  # 只向前传播
                        start_frame_idx=0,  # 从第0帧开始
                        max_frame_num_to_track=max_frames,  # 限制处理帧数
                    )
                ):
                    frame_idx = result_dict["frame_index"]
                    outputs = result_dict["outputs"]
                    all_outputs[frame_idx] = outputs
                    frame_count += 1
                    # 打印前3帧的调试信息
                    if frame_count <= 3:
                        print(f"    [调试] 帧{frame_idx}: outputs keys={list(outputs.keys())}")
                
                # 4. 统计结果
                all_object_ids = set()
                for outputs in all_outputs.values():
                    if "out_obj_ids" in outputs:
                        all_object_ids.update(outputs["out_obj_ids"])
                
                num_objects = len(all_object_ids)
                num_frames = len(all_outputs)
                
                # 调试：打印第一帧的outputs结构
                if 0 in all_outputs:
                    print(f"    [调试] 第0帧outputs键: {list(all_outputs[0].keys())}")
                    if 'out_obj_ids' in all_outputs[0]:
                        print(f"    [调试] out_obj_ids: {all_outputs[0]['out_obj_ids']}")
                    if 'out_binary_masks' in all_outputs[0]:
                        print(f"    [调试] out_binary_masks shape: {all_outputs[0]['out_binary_masks'].shape if hasattr(all_outputs[0]['out_binary_masks'], 'shape') else 'N/A'}")
                
                results_per_prompt[text_prompt] = {
                    'session_id': session_id,
                    'num_objects': num_objects,
                    'num_frames': num_frames,
                    'object_ids': list(all_object_ids),
                    'outputs': all_outputs  # 完整输出，包含masks等
                }
                
                print(f"    ✓ 检测到 {num_objects} 个对象")
                print(f"    ✓ 处理了 {num_frames} 帧")
                
                # 可视化：保存所有有对象的帧
                if num_objects > 0:
                    frames_with_objects = []
                    for frame_idx in all_outputs.keys():
                        if 'out_obj_ids' in all_outputs[frame_idx] and len(all_outputs[frame_idx]['out_obj_ids']) > 0:
                            frames_with_objects.append(frame_idx)
                    
                    if frames_with_objects:
                        print(f"    [可视化] 保存{len(frames_with_objects)}帧的分割mask")
                        self._save_all_frames_visualization(video_path, all_outputs, frames_with_objects, text_prompt)
                
            except Exception as e:
                print(f"    ⚠️ 分割失败: {e}")
                results_per_prompt[text_prompt] = {'error': str(e)}
        
        return results_per_prompt
    
    def _save_all_frames_visualization(self, video_path, all_outputs, frame_indices, prompt):
        """保存所有有对象的帧的可视化，使用ffmpeg提取正确亮度的帧"""
        import subprocess
        import tempfile
        
        vis_dir = './outputs/sam3_visualizations'
        os.makedirs(vis_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 使用临时目录存储ffmpeg提取的帧
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 用ffmpeg提取所有有对象的帧
            for frame_idx in frame_indices:  # 保存所有帧
                # 使用ffmpeg提取指定帧
                output_frame = os.path.join(tmp_dir, f"frame_{frame_idx}.jpg")
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vf', f'select=eq(n\\,{frame_idx})',
                    '-vframes', '1',
                    output_frame,
                    '-y', '-loglevel', 'quiet'
                ]
                subprocess.run(cmd, check=True)
                
                # 读取ffmpeg提取的帧
                frame = cv2.imread(output_frame)
                if frame is None:
                    continue
                
                # 获取该帧的masks
                outputs = all_outputs[frame_idx]
                if 'out_binary_masks' not in outputs or len(outputs['out_binary_masks']) == 0:
                    continue
                
                masks = outputs['out_binary_masks']
                
                # 为每个对象保存mask overlay
                for obj_idx, mask in enumerate(masks):
                    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                    
                    # 创建overlay：原始帧 + mask高亮
                    overlay = frame.copy()
                    color = np.array([0, 255, 0], dtype=np.float32)
                    mask_bool = mask_np > 0
                    overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)
                    
                    # 保存
                    output_path = os.path.join(vis_dir, f"{video_name}_frame{frame_idx}_{prompt}_obj{obj_idx}.jpg")
                    cv2.imwrite(output_path, overlay)
                
                print(f"      ✓ 帧{frame_idx}: {len(masks)}个对象")
    
    def _save_visualization(self, video_path, outputs, prompt, frame_idx):
        """保存指定帧的分割mask可视化"""
        try:
            # 读取指定帧
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"    ⚠️ 无法读取帧{frame_idx}")
                return
            
            # 不使用TV range转换，直接使用原始帧（因为转换后反而更暗）
            
            # 获取masks
            if 'out_binary_masks' not in outputs or len(outputs['out_binary_masks']) == 0:
                return
            
            masks = outputs['out_binary_masks']  # shape: (N, H, W) N个对象
            
            # 创建输出目录
            vis_dir = './outputs/sam3_visualizations'
            os.makedirs(vis_dir, exist_ok=True)
            
            # 为每个对象保存mask
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            for obj_idx, mask in enumerate(masks):
                # 转换mask为彩色overlay
                mask_np = mask.cpu().numpy() 
                
                # 创建可视化：背景激进提亮 + mask高亮
                # 先将原始帧提亮4倍以解决暗场景问题
                overlay = np.clip(frame.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
                color = np.array([0, 255, 0], dtype=np.float32)  # 绿色
                
                # 在mask区域叠加高亮颜色
                mask_bool = mask_np > 0
                overlay[mask_bool] = (overlay[mask_bool] * 0.4 + color * 0.6).astype(np.uint8)
                
                # 保存
                output_path = os.path.join(vis_dir, f"{video_name}_{prompt}_obj{obj_idx}.jpg")
                cv2.imwrite(output_path, overlay)
                print(f"    ✓ 保存可视化: {output_path}")
        except Exception as e:
            print(f"    ⚠️ 可视化保存失败: {e}")

def load_test_samples(data_path, num_samples=3):
    """加载测试样本"""
    with open(data_path) as f:
        data = json.load(f)
    
    # 获取数据目录（用于构建视频完整路径）
    data_dir = os.path.dirname(data_path)
    
    samples = []
    for item in data[:num_samples]:
        # 支持两种格式：video_path（完整路径）或 video（文件名）
        video_path = item.get('video_path', '')
        if not video_path:
            video_name = item.get('video', '')
            if video_name:
                # 尝试在数据目录下找视频
                video_path = os.path.join(data_dir, video_name)
        
        # 支持两种格式：question 或 problem + options
        question = item.get('question', '')
        if not question:
            problem = item.get('problem', '')
            options = item.get('options', [])
            if problem and options:
                options_text = '\n'.join(options)
                question = f"{problem} Options:\n{options_text}\nPlease provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        
        samples.append({
            'qid': item.get('qid', ''),
            'video_path': video_path,
            'question': question,
            'problem_type': item.get('problem_type', ''),
            'solution': item.get('solution', item.get('answer', ''))
        })
    
    return samples

def run_full_pipeline(args):
    """运行完整的pipeline"""
    
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
    
    # 3. SAM3 将在主模型推理完成后加载（避免显存冲突）
    print("="*80)
    print("Step 3: SAM3分割模型（将在推理完成后加载）")
    print("="*80)
    sam3 = None  # 延迟加载
    sam3_checkpoint_path = None
    if args.enable_sam3:
        sam3_checkpoint_path = os.path.join(args.sam3_model, "sam3.pt")
        if not os.path.exists(sam3_checkpoint_path):
            print(f"⚠️ 本地checkpoint不存在: {sam3_checkpoint_path}")
            sam3_checkpoint_path = None
        print("✓ SAM3 将在主模型释放显存后加载")
    else:
        print("✓ 跳过SAM3")
    print()
    
    # 4. 加载测试样本
    print("="*80)
    print(f"Step 4: 加载测试样本 (前{args.num_samples}个)")
    print("="*80)
    test_samples = load_test_samples(args.data_path, args.num_samples)
    print(f"✓ 加载了 {len(test_samples)} 个测试样本\n")
    
    # 5. 运行完整pipeline
    print("="*80)
    print("Step 5: 运行完整Pipeline")
    print("="*80)
    
    results = []
    
    for idx, sample in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"样本 {idx+1}/{len(test_samples)}")
        print(f"{'='*80}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Video: {sample['video_path']}")
        
        # 5.1 构建输入（限制视频帧数防止token过长）
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample['video_path'], "nframes": args.video_max_frames},
                    {"type": "text", "text": sample['question']}
                ]
            }
        ]
        
        print(f"  视频最大帧数: {args.video_max_frames}")
        audios, images, videos = process_mm_info(message, use_audio_in_video=False)
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        model_inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        
        # 5.2 生成thinking
        print("\n[1/3] 生成thinking...")
        with torch.inference_mode():
            text_ids = model.generate(
                **model_inputs.to(model.device).to(model.dtype),
                use_audio_in_video=False,
                max_new_tokens=2048
            )
        
        full_output = processor.decode(text_ids[0][model_inputs.input_ids.size(1):], skip_special_tokens=True)
        context = extract_context(full_output)
        thinking = extract_think(full_output)
        answer = extract_answer(full_output)
        
        print(f"✓ Context生成完成 (长度: {len(context)} 字符)")
        print(f"✓ Thinking生成完成 (长度: {len(thinking)} 字符)")
        print(f"✓ Answer: {answer}")
        
        # 5.3 总结thinking
        print("\n[2/3] 总结thinking...")
        summary = summarizer.summarize(thinking)
        print(f"✓ Summary生成完成 (长度: {len(summary)} 字符)")
        
        # 提取SAM3指令
        sam3_objects = parse_sam3_instructions(summary)
        print(f"✓ 提取SAM3分割对象: {sam3_objects}")
        
        # 简化prompt为SAM3友好的类别
        sam3_objects_simplified = []
        if sam3_objects:
            print("\n简化prompt:")
            for obj in sam3_objects:
                simplified = simplify_prompt_for_sam3(obj)
                sam3_objects_simplified.append(simplified)
                print(f"  '{obj}' -> '{simplified}'")
            # 去重
            sam3_objects_simplified = list(dict.fromkeys(sam3_objects_simplified))
            print(f"✓ 简化后对象: {sam3_objects_simplified}")
            
            # 为防止OOM，只处理第一个对象
            if len(sam3_objects_simplified) > 1:
                print(f"⚠️ 为防止OOM，仅处理第一个对象: '{sam3_objects_simplified[0]}'")
                sam3_objects_simplified = [sam3_objects_simplified[0]]
        
        # 5.4 暂存 SAM3 分割任务（稍后执行）
        sam3_results = None
        if sam3_objects_simplified and args.enable_sam3:
            print(f"\n[3/3] SAM3分割任务已记录（将在主模型释放后执行）")
        else:
            print(f"\n[3/3] 跳过SAM3分割 (enable_sam3={args.enable_sam3}, has_objects={bool(sam3_objects)})")
        
        # 保存结果
        result = {
            'sample_id': idx + 1,
            'qid': sample['qid'],
            'question': sample['question'],
            'video_path': sample['video_path'],
            'ground_truth': sample['solution'],
            'full_output': full_output,  # 完整模型输出（含<context>、<think>和<answer>标签）
            'context': context,
            'thinking': thinking,
            'answer': answer,
            'summary': summary,
            'sam3_objects': sam3_objects,
            'sam3_results_summary': {
                prompt: {
                    'num_objects': data.get('num_objects', 0),
                    'num_frames': data.get('num_frames', 0),
                    'object_ids': data.get('object_ids', []),
                    'error': data.get('error')
                }
                for prompt, data in (sam3_results or {}).items()
            } if sam3_results else None
        }
        results.append(result)
        
        print(f"\n✓ 样本 {idx+1} 处理完成")
    
    # 6. 释放主模型显存，执行 SAM3 分割
    if args.enable_sam3 and sam3_checkpoint_path:
        print("\n" + "="*80)
        print("Step 6: 释放主模型显存，加载SAM3执行分割")
        print("="*80)
        
        # 释放 HumanOmniV2 和 Summarizer 显存
        print("释放主模型显存...")
        del model
        del processor
        del summarizer
        import gc
        gc.collect()
        
        # 彻底清理 CUDA 状态
        print("彻底清理CUDA状态...")
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # 重置默认设备到 SAM3 的 GPU
        torch.cuda.set_device(args.sam3_gpu)
        print(f"✓ 主模型显存已释放，CUDA 默认设备设为 GPU {args.sam3_gpu}")
        
        # 加载 SAM3
        print(f"\n加载SAM3模型 (GPU {args.sam3_gpu})...")
        sam3 = SAM3Segmenter(checkpoint_path=sam3_checkpoint_path, gpu_id=args.sam3_gpu)
        
        # 执行所有样本的 SAM3 分割
        print("\n执行SAM3分割任务...")
        for idx, result in enumerate(results):
            sam3_objects = result.get('sam3_objects', [])
            if not sam3_objects:
                continue
            
            # 简化 prompt
            sam3_objects_simplified = []
            for obj in sam3_objects:
                simplified = simplify_prompt_for_sam3(obj)
                sam3_objects_simplified.append(simplified)
            sam3_objects_simplified = list(dict.fromkeys(sam3_objects_simplified))
            
            # 只处理第一个对象
            if len(sam3_objects_simplified) > 1:
                sam3_objects_simplified = [sam3_objects_simplified[0]]
            
            print(f"\n处理样本 {idx+1}: {sam3_objects_simplified}")
            try:
                sam3_results = sam3.segment_video(
                    result['video_path'],
                    sam3_objects_simplified,
                    max_frames=args.max_frames
                )
                # 更新结果
                results[idx]['sam3_results_summary'] = {
                    prompt: {
                        'num_objects': data.get('num_objects', 0),
                        'num_frames': data.get('num_frames', 0),
                        'object_ids': data.get('object_ids', []),
                        'error': data.get('error')
                    }
                    for prompt, data in (sam3_results or {}).items()
                }
            except Exception as e:
                print(f"⚠️ SAM3分割失败: {e}")
                results[idx]['sam3_results_summary'] = {'error': str(e)}
        
        print("\n✓ SAM3分割完成")
    
    # 7. 保存结果（移除不能JSON序列化的outputs字段）
    output_file = os.path.join(args.output_dir, "full_pipeline_results.json")
    
    # 清理results，移除outputs中的masks等大型数据
    cleaned_results = []
    for result in results:
        cleaned_result = result.copy()
        if 'sam3_results_summary' in cleaned_result and cleaned_result['sam3_results_summary']:
            cleaned_summary = {}
            for prompt, data in cleaned_result['sam3_results_summary'].items():
                cleaned_data = {
                    'num_objects': int(data.get('num_objects', 0)),
                    'num_frames': int(data.get('num_frames', 0)),
                    'object_ids': [int(x) for x in data.get('object_ids', [])],
                    'error': data.get('error')
                }
                cleaned_summary[prompt] = cleaned_data
            cleaned_result['sam3_results_summary'] = cleaned_summary
        cleaned_results.append(cleaned_result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("Pipeline完成！")
    print(f"{'='*80}")
    print(f"✓ 结果已保存到: {output_file}")
    print(f"✓ 处理了 {len(results)} 个样本")

def main():
    parser = argparse.ArgumentParser(description="完整Pipeline: HumanOmniV2 + Summarizer + SAM3")
    
    # HumanOmniV2模型
    parser.add_argument('--humanomniv2-model', type=str, 
                        default='os.environ.get('MODEL_CHECKPOINT_DIR', './checkpoints')/stage5_outcome_reward/checkpoint-1083')
    
    # Summarizer模型
    parser.add_argument('--base-model', type=str,
                        default='${PROJECT_ROOT}/Qwen2.5-3B-Instruct')
    parser.add_argument('--lora-model', type=str,
                        default='${PROJECT_ROOT}/thinking_summarizer/outputs/thinking_summarizer_6770/final_model')
    
    # SAM3模型
    parser.add_argument('--sam3-model', type=str,
                        default='os.environ.get('SAM3_CHECKPOINT_DIR', './sam3/checkpoints')')
    parser.add_argument('--enable-sam3', action='store_true', default=False,
                        help='是否启用SAM3分割')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='SAM3处理的最大帧数（默认处理所有帧）')
    parser.add_argument('--sam3-gpu', type=int, default=1,
                        help='SAM3使用的GPU ID（默认1）')
    
    # 视频处理
    parser.add_argument('--video-max-frames', type=int, default=64,
                        help='HumanOmniV2处理视频的最大帧数（默认64，防止token过长）')
    
    # 数据
    parser.add_argument('--data-path', type=str,
                        default='${PROJECT_ROOT}/thinking_summarizer/data/my_demo/demo_test_data.json')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='测试样本数量')
    
    # 输出
    parser.add_argument('--output-dir', type=str,
                        default='./outputs/full_pipeline')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否可视化SAM3结果')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_full_pipeline(args)

if __name__ == '__main__':
    main()
