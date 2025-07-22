import torch

from ops_utils import *

# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    # torch.arange(0, 10000, device='tcu') # Could not run 'aten::arange.start_out' with arguments from the 'tcu' backend.
    # torch.arange(517,device='tcu')
    input_cpu = torch.arange(0, 2, 0.5) # Could not run 'aten::arange.start_out' with arguments from the 'tcu' backend.
    out_tcu = torch.arange(0, 2, 0.5, device='tcu')
    out_tcu_to_cpu = out_tcu.to('cpu')
    ret = torch.allclose(out_tcu_to_cpu, input_cpu, 1e-5, 1e-8)
    assert (ret is True)

    input_cpu = torch.arange(0, 10000)
    out_tcu = torch.arange(0, 10000, device='tcu')
    out_tcu_to_cpu = out_tcu.to('cpu')
    ret = torch.allclose(out_tcu_to_cpu, input_cpu, 1e-5, 1e-8)
    assert (ret is True)

    print("PASSED")
