#!/usr/bin/env python3
"""
测试训练好的LoRA模型
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 配置
BASE_MODEL = "${PROJECT_ROOT}/Qwen2.5-3B-Instruct"
LORA_MODEL = "${PROJECT_ROOT}/thinking_summarizer/outputs/thinking_summarizer_6770/final_model"
TEST_DATA = "${PROJECT_ROOT}/thinking_summarizer/data/training_dataset_6770/val.json"

print("=" * 80)
print("测试LoRA模型 - Thinking Summarizer")
print("=" * 80)
print(f"基座模型: {BASE_MODEL}")
print(f"LoRA模型: {LORA_MODEL}")
print("=" * 80)

# 加载模型
print("\n1. 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_MODEL)
model.eval()
print("✓ 模型加载完成")

# 加载测试数据
print("\n2. 加载测试数据...")
with open(TEST_DATA, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
print(f"✓ 加载 {len(test_data)} 条验证数据")

# 测试函数
def generate_summary(thinking_text, instruction):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction}\n\nThinking Process:\n{thinking_text}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
    
    summary = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return summary

# 测试3个样本
print("\n3. 测试模型输出...")
print("=" * 80)

for i in range(min(3, len(test_data))):
    sample = test_data[i]
    thinking = sample['input']
    instruction = sample['instruction']
    expected = sample['output']
    
    print(f"\n【测试样本 {i+1}】")
    print(f"Video ID: {sample['metadata'].get('video_id', 'N/A')}")
    print(f"来源: {sample['metadata'].get('source', 'N/A')}")
    print(f"\nThinking (前200字):")
    print(f"  {thinking[:200]}...")
    
    # 生成summary
    generated = generate_summary(thinking, instruction)
    
    print(f"\n生成的Summary:")
    print("-" * 40)
    print(generated[:800])
    print("-" * 40)
    
    print(f"\n期望的Summary (前300字):")
    print(f"  {expected[:300]}...")
    print("=" * 80)

print("\n✓ 测试完成！")
