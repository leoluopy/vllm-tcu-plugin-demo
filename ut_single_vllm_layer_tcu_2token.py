import sys,os

import torch
import torch_tcu
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

import os
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

from vllm import LLM, SamplingParams

# 1. 初始化模型参数
# model_path = "/root/DeepSeek-R1-Distill-Llama-8B"  # 替换为实际权重路径
model_path = "/home/leo/Downloads/pretrained_models/DeepSeek-R1-Distill-Llama-8B/"
sampling_params = SamplingParams(
    temperature=0.7,      # 控制生成随机性（0-1，越大越有创意）
    top_p=0.95,           # 核采样参数
    max_tokens=2,       # 最大生成token数
    stop=["<|eot_id|>"]   # Llama3的终止符[6,9](@ref)
)

# 2. 加载模型（自动检测可用GPU）
llm = LLM(
    model=model_path,
    max_model_len=8192,  # 显式限制最大长度
    tensor_parallel_size=1,
    device='tcu',
    dtype="auto",
    # dtype="float16",
    gpu_memory_utilization=0.93  # 适当提升显存利用率
)

# 3. 构造对话prompt（Llama3的指令格式）
prompt = '''<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
你是一个有帮助的AI助手<|eot_id|>
<|start_header_id|>user<|end_header_id|>
中国首都在哪里？请不要输出</think>的过程，并且你只能输出两个token <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>'''

# 4. 执行推理
outputs = llm.generate([prompt], sampling_params)
print('\n\n#######\n+{}'.format(outputs))
# 5. 处理输出结果
for output in outputs:
    generated_text = output.outputs[0].text
    # 清理特殊token并格式化输出
    response = generated_text.split("<|eot_id|>")[0].strip()
    print("模型回复：\n", response)
    print("len:{}".format(len(response)))