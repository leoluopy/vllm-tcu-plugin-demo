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
    gpu_memory_utilization=0.93 , # 适当提升显存利用率
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

# +[RequestOutput(request_id=0, prompt='<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>
# \n你是一个有帮助的AI助手<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n中国首都在哪里？请不要输出</think>的过程，并且你只能输出两个token
# <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>', prompt_token_ids=[128000, 27, 91, 7413, 3659, 4424, 91, 397, 128006, 9125,
# 128007, 198, 57668, 122503, 19361, 123725, 9554, 15836, 103129, 46034, 128009, 198, 128006, 882, 128007, 198, 59795, 61075, 72368, 19000,
# 125011, 11571, 15225, 113473, 67117, 128014, 9554, 112696, 91495, 103786, 57668, 122332, 67117, 110835, 5963, 220, 128009, 198, 128006,
# 78191, 128007], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None,
# outputs=[CompletionOutput(index=0, text='hu!', token_ids=(17156, 0), cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1749797930.0902662, last_token_time=1749798041.585685, first_scheduled_time=1749797930.0917509, first_token_time=1749798031.5905726, time_in_queue=0.0014846324920654297, finished_time=1749798041.6088455, scheduler_time=0.05354313900170382, model_forward_time=None, model_execute_time=None, spec_token_acceptance_counts=[0]), lora_request=None, num_cached_tokens=0, multi_modal_placeholders={})]
# 模型回复：
#  hu!
# len:3