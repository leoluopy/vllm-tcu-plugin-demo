import torch.nn.functional as F
from ops_utils import *

# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    large_logits = torch.randn(10000, 5000) * 100  # 10000样本 x 5000类别
    output_gt = F.softmax(large_logits, dim=1)
    large_logits_tcu = large_logits.to(device)
    output_tcu = F.softmax(large_logits_tcu, dim=1)
    output_tcu_to_cpu = output_tcu.to('cpu')
    assert (torch.allclose(output_gt, output_tcu_to_cpu, 1e-5, 1e-8) is True)

    extreme_logits = torch.zeros(100, 10000)
    extreme_logits[:, 0] = 1e10    # 第一个类别极大值
    extreme_logits[:, 1] = -1e10   # 第二个类别极小值
    output_gt = F.softmax(extreme_logits, dim=1)

    extreme_logits_tcu = extreme_logits.to(device)
    output_tcu = F.softmax(extreme_logits_tcu, dim=1)
    output_tcu_to_cpu = output_tcu.to('cpu')
    assert (torch.allclose(output_gt, output_tcu_to_cpu, 1e-5, 1e-8) is True)


    seq_logits = torch.randn(32, 100000)  # batch_size=32, seq_len=100K
    output_gt = F.softmax(seq_logits, dim=1)

    seq_logits_tcu = seq_logits.to(device)
    output_tcu = F.softmax(seq_logits_tcu, dim=1)
    output_tcu_tcu_to_cpu = output_tcu.to('cpu')
    assert (torch.allclose(output_gt, output_tcu_tcu_to_cpu, 1e-5, 1e-8) is True)

    print("PASSED")
