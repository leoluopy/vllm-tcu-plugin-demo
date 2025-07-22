from ops_utils import *

# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    x = torch.arange(16).reshape(4, 4)
    y = x.view(16)
    x_tcu = x.to(device)
    y_tcu = x_tcu.view(16)
    y_tcu_to_cpu = y_tcu.to('cpu')
    assert (torch.allclose(y, y_tcu_to_cpu, 1e-5, 1e-8) is True)

    x = torch.randn(3, 4, 5)
    y = x.view(3, -1)
    x_tcu = x.to(device)
    y_tcu = x_tcu.view(3, -1)
    y_tcu_to_cpu = y_tcu.to('cpu')
    assert (torch.allclose(y, y_tcu_to_cpu, 1e-5, 1e-8) is True)

    x = torch.empty(0)
    y = x.view(1, 0)
    x_tcu = x.to(device)
    y_tcu = x_tcu.view(1, 0)
    y_tcu_to_cpu = y_tcu.to('cpu')
    assert (torch.allclose(y, y_tcu_to_cpu, 1e-5, 1e-8) is True)

    print("PASSED")
