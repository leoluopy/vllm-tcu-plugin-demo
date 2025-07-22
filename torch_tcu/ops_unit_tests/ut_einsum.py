import math

import torch
import torch_tcu
from ops_utils import *

# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    # 测试用例 6: 外积 (等价于 torch.outer)
    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = torch.tensor([4, 5, 6], dtype=torch.float32, device=device)
    einsum_result = torch.einsum('i,j->ij', a, b)
    einsum_result_cpu = einsum_result.to('cpu')
    torch_result = torch.outer(a.cpu(), b.cpu())
    assert torch.allclose(einsum_result_cpu, torch_result), "外积测试失败"
