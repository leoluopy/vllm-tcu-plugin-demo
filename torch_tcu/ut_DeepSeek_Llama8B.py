import torch
import torch_tcu
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

# device = "cuda"
device = "tcu"

# 加载模型和分词器
# model_name = "/root/DeepSeek-R1-Distill-Llama-8B/"
model_name = "/home/leo/Downloads/pretrained_models/DeepSeek-R1-Distill-Llama-8B/"  # 实际使用时替换为正确的8B模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 使用bfloat16减少显存占用
    device_map="auto",          # 自动分配多GPU资源
    low_cpu_mem_usage=True
).to(device).eval()

# 生成参数配置
generation_config = {
    "max_new_tokens": 128,      # 最大生成token数
    "temperature": 0.6,         # 控制随机性 (0.0-1.0)
    "top_p": 0.9,              # Nucleus采样参数
    "do_sample": True,         # 启用采样模式
    "pad_token_id": tokenizer.eos_token_id  # 设置padding token
}

# 推理函数
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]  # 返回生成的文本部分

# 示例使用
if __name__ == "__main__":
    prompt = "人工智能的未来发展方向是"
    result = generate_text(prompt)
    print(f"Input: {prompt}\nOutput: {result}")