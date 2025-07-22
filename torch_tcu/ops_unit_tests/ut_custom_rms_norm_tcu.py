import torch
import torch_tcu  # 自定义TCU后端实现

# 重命名并注册设备模块
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)

# 生成标准张量方法
torch.utils.generate_methods_for_privateuse1_backend()

# torch.ops.load_library("build/lib.linux-x86_64-cpython-310/torch_tcu.cpython-310-x86_64-linux-gnu.so")


device = 'tcu'
# device = 'cuda'

# 调用注册的算子
input = torch.ones(32, 512,device=device)
weight = torch.ones(512,device=device)
# ref invoke: torch.ops._C.rms_norm(out, input, weight, epsilon)
output = torch.ops._C.rms_norm(input, weight, eps=1e-6)

# print(output.shape)
print("PASSED")

