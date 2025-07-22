import torch
import vllm_tcu._C


def rms_norm(out: torch.Tensor, intpu: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)




