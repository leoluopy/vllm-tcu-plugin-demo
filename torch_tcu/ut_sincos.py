import torch
import torch_tcu

torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

if __name__ == '__main__':

    x1 = torch.randn([51,32,64],device='tcu',dtype=torch.float16)
    x2 = torch.randn([51,32,64],device='tcu',dtype=torch.float16)

    sin = torch.randn([51,1,64],device='tcu',dtype=torch.float16)
    cos = torch.randn([51,1,64],device='tcu',dtype=torch.float16)

    o1 = x1 * cos - x2 * sin
    # o2 = x2 * cos + x1 * sin
    o2_1 = x2 * cos
    o2_2 = x1 * sin
    o2 = o2_1 + o2_2

    print("END")


