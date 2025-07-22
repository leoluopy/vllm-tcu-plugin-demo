import torch
import torch_tcu
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

x1 = torch.ones(4, 4, device='tcu')
x1_cpu = x1.to('cpu')

x2 = torch.empty(3, 4)
x2.tcu()

