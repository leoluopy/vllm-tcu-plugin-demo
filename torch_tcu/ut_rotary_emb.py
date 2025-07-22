import torch
import torch_tcu
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

torch.utils.rename_privateuse1_backend("tcu")
torch._register_device_module("tcu", torch_tcu)
torch.utils.generate_methods_for_privateuse1_backend()

if __name__ == '__main__':

    batch_size, seq_len, num_heads = 2, 3, 4
    # device = 'cpu'
    device = 'tcu'

    head_size = 64
    base = 10000
    rotary_emb = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=head_size,
        base=base,
        max_position_embeddings=2048,
        is_neox_style=False,
        dtype=torch.float16,
    )

    shape = (batch_size, seq_len, num_heads, head_size)

    # 创建随机输入
    torch.manual_seed(42)
    query = torch.randn(shape, dtype=torch.float16, device=device)
    key = torch.randn(shape, dtype=torch.float16, device=device)

    # 位置IDs (0 到 seq_len-1)
    positions = torch.arange(seq_len, dtype=torch.long, device=device)

    # 应用Rotary Embedding
    rotated_query, rotated_key = rotary_emb(positions, query, key)


    print("END")
