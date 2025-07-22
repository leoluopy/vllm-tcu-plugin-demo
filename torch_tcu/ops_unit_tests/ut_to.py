
from ops_utils import *


# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    lh_gt = torch.rand([64, 512, 64], dtype=torch.float16)
    lh_tcu = lh_gt.to(device)
    out_tcu_to_cpu = lh_tcu.to('cpu')

    log_errors(out_tcu_to_cpu, lh_gt)

    check_ret = torch.allclose(out_tcu_to_cpu, lh_gt, 1e-5, 1e-8)
    print("check_ret:{}".format(check_ret))
    print("END")
