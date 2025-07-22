import math

import torch
import torch_tcu
from ops_utils import *

# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    param_data = torch.randn(28672, 4096, device=device)

    # 设置测试参数
    output_dim = 0
    shard_offset = 0
    shard_size = 14336

    print("param_data before narrowed : {} stride:{}".format(param_data.shape,param_data.stride()))
    # # 执行 narrow 操作
    narrowed_tensor = param_data.narrow(output_dim, shard_offset, shard_size)
    print("narrowed_tensor shapes: {} stride:{}".format(narrowed_tensor.shape,narrowed_tensor.stride()))
