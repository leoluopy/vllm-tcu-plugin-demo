import torch
import torch_tcu
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

if __name__ == '__main__':

    x = torch.randn([51,4096],device='tcu')
    weight = torch.randn([6144,4096],device='tcu')

    out = torch.nn.functional.linear(x,weight,None)

    print("END")

