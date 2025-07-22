import torch
import torch_tcu
torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

def log_errors(output_tcu_copied_cpu, output_cpu):
    errors = (output_tcu_copied_cpu - output_cpu).abs()

    error_num = (errors > 1e-2).int().sum()
    whole_num = errors.numel()
    print("error max: {}, mean:{}, >1e-2 {}/{}={}".format(errors.max(), errors.mean(), error_num, whole_num,
                                                          error_num / whole_num))
    check_ret = torch.allclose(output_tcu_copied_cpu, output_cpu, 1e-5, 1e-8)
    return check_ret


