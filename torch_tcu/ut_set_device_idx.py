import torch
import torch_tcu

torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

if __name__ == '__main__':

    # 验证 set_device 功能
    torch.tcu.set_device(torch.device('tcu',0))
    default_device = torch.tcu.get_device()
    print("default_device: {}".format(default_device))
    x = torch.randn(2, 2)  # 默认在 CPU
    y = x.to('tcu')        # 显式迁移到 TCU
    assert y.device.type == 'tcu'
    assert y.device.index == 0  # 确认设备索引

    # 切换设备
    torch.tcu.set_device(torch.device('tcu',1))
    default_device = torch.tcu.get_device()
    print("default_device: {}".format(default_device))
    z = torch.tensor([1,2,3], device='tcu')
    assert z.device.index == 1  # 确认设备索引

    print("PASSED")