import torch
import torch_tcu

from ops_unit_tests.ops_utils import log_errors
import torchvision.models

if __name__ == '__main__':
    res50 = torchvision.models.resnet50()
    res50.eval()
    x = torch.ones([1, 3, 224, 224])
    y = res50(x)

    res50_tcu = res50.to('tcu')
    x_tcu = x.to('tcu')
    out_tcu = res50_tcu(x_tcu)
    out_tcu_to_cpu = out_tcu.to('cpu')

    check_ret = log_errors(out_tcu_to_cpu, y)
    assert (check_ret, True)
    print("PASSED")
