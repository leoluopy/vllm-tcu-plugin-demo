import torch
import vllm_tcu._C

def test_rms_norm(device):
    input = torch.ones(32, 512,device=device)
    weight = torch.ones(512,device=device)
    # ref invoke: torch.ops._C.rms_norm(out, input, weight, epsilon)
    output_tcu = torch.ops._C.rms_norm(input, weight, eps=1e-6)



def test_reshape_and_cache(device):
    key = torch.randn([51,8,128],device=device,dtype=torch.float16)
    value = torch.randn([51,8,128],device=device,dtype=torch.float16)
    key_cache = torch.randn([378,128,8,128],device=device,dtype=torch.float16)
    value_cache = torch.randn([378,128,8,128],device=device,dtype=torch.float16)
    slots = torch.randn([51],device=device,dtype=torch.float16)

    torch.ops._C.reshape_and_cache(key=key,
                                   value=value,
                                   key_cache=key_cache,
                                   value_cache=value_cache,
                                   slot_indices=slots)
    print("reshape and cache invoked")

def test_sdp(device):
    query = torch.randn([51,32,128],device=device,dtype=torch.bfloat16)
    key = torch.randn([51,8,128],device=device,dtype=torch.bfloat16)
    value = torch.randn([51,8,128],device=device,dtype=torch.bfloat16)
    mask = torch.randn([51,51],device=device,dtype=torch.bfloat16)
    scale = 0.08838834764831845

    print("    $$$$$$$$$$$$$$$$$$$$$$$ attension to invoke ")
    output = torch.ops._C.flash_attention_qlens(
        query=query,
        key=key,
        value=value,
        mask=mask,
        scale_value=scale)
    print(" attention output shape:{}".format(output.shape))

if __name__ == '__main__':

    import torch
    import torch_tcu  # 自定义TCU后端实现

    # 重命名并注册设备模块
    torch.utils.rename_privateuse1_backend("tcu")
    torch._register_device_module("tcu", torch_tcu)

    # 生成标准张量方法
    torch.utils.generate_methods_for_privateuse1_backend()

    # torch.ops.load_library("build/lib.linux-x86_64-cpython-310/torch_tcu.cpython-310-x86_64-linux-gnu.so")


    device = 'tcu'
    # device = 'cpu'

    # 调用注册的算子
    # test_rms_norm(device)
    # test_reshape_and_cache(device)
    test_sdp(device)

    # print(output.shape)
    print("PASSED")


