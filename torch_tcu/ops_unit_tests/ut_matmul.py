
from ops_utils import *

# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    lh = torch.rand([64, 512, 64], dtype=torch.float16)
    rh = torch.rand([64, 64, 512], dtype=torch.float16)
    # lh = torch.ones([64, 512, 64], dtype=torch.float16) * 2
    # rh = torch.ones([64, 64, 512], dtype=torch.float16) * 3

    # lh = torch.rand([8, 64, 32], dtype=torch.float16)
    # rh = torch.rand([8, 32, 5], dtype=torch.float16)

    out_gt = torch.matmul(lh, rh)
    print(out_gt.shape)

    lh_tcu = lh.to(device)
    rh_tcu = rh.to(device)
    out_tcu = torch.matmul(lh_tcu, rh_tcu)
    print(out_tcu.shape)
    out_tcu_to_cpu = out_tcu.to('cpu')

    log_errors(out_tcu_to_cpu, out_gt)

    check_ret = torch.allclose(out_tcu_to_cpu, out_gt, 1e-5, 1e-8)
    print("check_ret:{}".format(check_ret))
    print("END")
