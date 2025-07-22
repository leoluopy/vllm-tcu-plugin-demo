import os

import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import threading

import torch_tcu

torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

# device = 'cuda'
device = 'tcu'


class Attention(nn.Module):
    def __init__(self, max_seq_len, head_dim, flash):
        super().__init__()
        self.flash = flash
        self.dropout = 0
        self.attn_dropout = nn.Dropout(self.dropout)
        self.head_dim = head_dim
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"), dtype=torch.half)
            mask = torch.triu(mask, diagonal=1).half()
            self.register_buffer("mask", mask)

    def forward(
            self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor):
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv,
                                                                      attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            # _xk = xk.clone()
            t = xk.transpose(2, 3)
            scores = torch.matmul(xq, t)
            scores = scores / math.sqrt(self.head_dim)
            a = self.mask[:, :, :seqlen, :seqlen]
            scores = scores + a
            scores = F.softmax(scores, dim=-1)
            scores = scores.type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)
        return output


def main(flash, bs, n_local_heads, seqlen, head_dim):
    torch.random.manual_seed(1)

    q = torch.ones((bs, n_local_heads, seqlen, head_dim), dtype=torch.float32).half()
    k = torch.ones((bs, n_local_heads, seqlen, head_dim), dtype=torch.float32).half()
    v = torch.ones((bs, n_local_heads, seqlen, head_dim), dtype=torch.float32).half()

    q.data.normal_(0, 0.1)
    k.data.normal_(0, 0.1)
    v.data.normal_(0, 0.1)

    model_cpu = Attention(seqlen, head_dim, flash).half()
    output_cpu = model_cpu(q, k, v)
    print("output_cpu shape:{}".format(output_cpu.shape))

    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    model_tcu = Attention(seqlen, head_dim, flash).half().to(device)
    output_tcu = model_tcu(q, k, v)
    output_tcu_copied_cpu = output_tcu.to('cpu')
    print("output_tcu shape:{}".format(output_tcu_copied_cpu.shape))

    assert (output_cpu.shape == output_tcu.shape)
    errors = (output_tcu_copied_cpu - output_cpu).abs()

    error_num = (errors > 1e-5).int().sum()
    whole_num = errors.numel()
    print("error max: {}, mean:{}, >1e-5 {}/{}={}".format(errors.max(), errors.mean(), error_num, whole_num,
                                                          error_num / whole_num))


if __name__ == '__main__':
    bs, n_local_heads, seqlen, head_dim = 8, 8, 512, 64
    main(False, bs, n_local_heads, seqlen, head_dim)
    print("END")
