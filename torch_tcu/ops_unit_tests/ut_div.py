import math

import torch
import torch_tcu
from ops_utils import *



# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    scores = torch.rand([8 , 8 , 512 , 512], dtype=torch.float16)
    out_gt = scores / math.sqrt(64)

    scores_tcu = scores.to(device)
    out_tcu = scores_tcu / math.sqrt(64)
    out_tcu_to_cpu = out_tcu.to('cpu')

    log_errors(out_tcu_to_cpu, out_gt)
    check_ret = torch.allclose(out_tcu_to_cpu, out_gt, 1e-5, 1e-8)
    print("check_ret:{}".format(check_ret))
    print("END")
