
from ops_utils import *

# device = 'cuda'
device = 'tcu'

if __name__ == '__main__':
    input_cpu = torch.randn(28672, 4096)
    print("input_cpu stride: {}".format(input_cpu.stride()))
    # 创建视图: 转置为(4,3), 步幅(1,4)
    size_d1=14336
    # size_d1=28672
    out_cpu_gt = input_cpu.as_strided(size=(size_d1, 4096), stride=(4096, 1))
    input_gcu = input_cpu.to(device)
    out_tcu = input_gcu.as_strided(size=(size_d1, 4096), stride=(4096, 1))
    out_tcu_to_cpu = out_tcu.to('cpu')
    ret = torch.allclose(out_tcu_to_cpu, out_cpu_gt, 1e-5, 1e-8)
    assert (ret is True)

    input_cpu = torch.arange(12).reshape(3, 4)
    print("input_cpu stride: {}".format(input_cpu.stride()))
    # 创建视图: 转置为(4,3), 步幅(1,4)
    out_cpu_gt = input_cpu.as_strided(size=(4, 3), stride=(1, 4))
    input_gcu = input_cpu.to(device)
    out_tcu = input_gcu.as_strided(size=(4, 3), stride=(1, 4))
    out_tcu_to_cpu = out_tcu.to('cpu')
    ret = torch.allclose(out_tcu_to_cpu, out_cpu_gt, 1e-5, 1e-8)
    assert (ret is True)

    input_cpu = torch.arange(12).reshape(3, 4)
    print("input_cpu stride: {}".format(input_cpu.stride()))
    # 创建视图: 转置为(4,3), 步幅(1,4)
    out_cpu_gt = input_cpu.as_strided(size=(4, 3), stride=(1, 4))
    input_gcu = input_cpu.to(device)
    out_tcu = input_gcu.as_strided(size=(4, 3), stride=(1, 4))
    out_tcu_to_cpu = out_tcu.to('cpu')
    ret = torch.allclose(out_tcu_to_cpu, out_cpu_gt, 1e-5, 1e-8)
    assert (ret is True)


    lh = torch.rand([64, 512, 64], dtype=torch.float16)
    print("input_cpu stride: {}".format(lh.stride()))
    lh_cpu_gt = lh.as_strided(size=(512, 64, 64), stride=(1, 512, 512 * 64))
    lh_gcu = lh.to(device)
    lh_tcu  = lh_gcu.as_strided(size=(512, 64, 64), stride=(1, 512, 512 * 64))
    lh_tcu_to_cpu = lh_tcu.to('cpu')
    ret = torch.allclose(lh_tcu_to_cpu, lh_cpu_gt, 1e-5, 1e-8)
    assert (ret is True)

    print("PASSED")

# log will be :
# input_cpu stride: (4, 1)
# as_strided input Tensor dimensions: 3 x 4
# [4, 3]
# [1, 4]
# self.is_contiguous():1
# as_strided tcuout Tensor dimensions: 4 x 3
# tcu_out.is_contiguous():1
# input_cpu stride: (4, 1)
# as_strided input Tensor dimensions: 3 x 4
# [4, 3]
# [1, 4]
# self.is_contiguous():1
# as_strided tcuout Tensor dimensions: 4 x 3
# tcu_out.is_contiguous():1
# input_cpu stride: (32768, 64, 1)
# as_strided input Tensor dimensions: 64 x 512 x 64
# [512, 64, 64]
# [1, 512, 32768]
# self.is_contiguous():1
# as_strided tcuout Tensor dimensions: 512 x 64 x 64
# tcu_out.is_contiguous():1
# PASSED