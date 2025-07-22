import torch
import torch.nn.functional as F


def paged_attention(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        num_heads: int,
        scale_value: float,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor
):
    """
    参数说明:
    query:        [num_seqs, num_heads, head_size]
    key_cache:     [num_blocks, num_kv_heads, block_size, head_size]
    value_cache:   [num_blocks, num_kv_heads, block_size, head_size]
    block_table:   [num_seqs, max_blocks_per_seq] 块ID映射表
    context_lens:  [num_seqs] 每个序列的实际长度
    out:           [num_seqs, num_heads, head_size] 输出张量
    """
    print(f"ATTENTION:page shapes:query:{query.shape},key_cache:{key_cache.shape},value_cache:{value_cache.shape},"
          f"num_kv_heads:{num_kv_heads},num_heads:{num_heads},scale_value:{scale_value},"
          f"block_table:{block_table.shape},context_lens:{context_lens.shape},out:{out.shape}");
    device = query.device
    dtype = query.dtype
    num_seqs = query.shape[0]
    head_size = query.shape[2]
    block_size = key_cache.shape[2]

    # 处理每个序列
    for seq_idx in range(num_seqs):
        # 获取当前序列信息
        seq_query = query[seq_idx]  # [num_heads, head_size]
        seq_context_len = context_lens[seq_idx].item()
        seq_block_ids = block_table[seq_idx]  # [max_blocks_per_seq]

        # 计算需要加载的块数量
        num_blocks = (seq_context_len + block_size - 1) // block_size

        # 预分配当前序列的键/值缓存
        valid_blocks = seq_block_ids[:num_blocks]
        seq_keys = key_cache[valid_blocks]  # [num_blocks, num_kv_heads, block_size, head_size]
        seq_values = value_cache[valid_blocks]

        # 处理块内实际有效长度
        last_block_len = seq_context_len % block_size
        if last_block_len == 0:
            last_block_len = block_size

        # 重组键值对张量
        keys = seq_keys.reshape(-1, num_kv_heads, head_size)[:seq_context_len]
        values = seq_values.reshape(-1, num_kv_heads, head_size)[:seq_context_len]

        # ================== 注意力核心计算 ==================
        # 处理分组查询注意力的头数不匹配问题
        if num_kv_heads != num_heads:
            rep_factor = num_heads // num_kv_heads
            keys = torch.repeat_interleave(keys, rep_factor, dim=1)
            values = torch.repeat_interleave(values, rep_factor, dim=1)

        # 重组张量维度:
        # keys: [seq_context_len, num_heads, head_size] -> [num_heads, seq_context_len, head_size]
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        seq_query = seq_query.unsqueeze(1)  # [num_heads, 1, head_size]

        # 计算注意力分数
        attn_scores = torch.matmul(seq_query, keys.transpose(-2, -1)) * scale_value
        attn_scores = attn_scores.squeeze(1)  # [num_heads, seq_context_len]

        # 应用softmax获取注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)  # [num_heads, seq_context_len]
        attn_probs = attn_probs.unsqueeze(1)  # [num_heads, 1, seq_context_len]

        # 计算注意力输出
        attn_output = torch.matmul(attn_probs, values)  # [num_heads, 1, head_size]
        attn_output = attn_output.squeeze(1)  # [num_heads, head_size]
        # ===================================================

        # 将结果写入输出张量
        out[seq_idx] = attn_output


# 使用示例
if __name__ == "__main__":
    # 配置参数
    num_seqs = 1
    num_heads = 32
    head_size = 128


    num_kv_heads = 8
    block_size = 128
    num_blocks = 940

    # 创建模拟输入
    query = torch.randn(num_seqs, num_heads, head_size)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size)
    block_table = torch.randint(0, num_blocks, (num_seqs, 10))  # 10 blocks max per seq
    context_lens = torch.randint(10, 300, (num_seqs,))  # 序列长度在10-300之间
    output = torch.zeros_like(query)

    # 调用paged attention
    paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=1.0 / (head_size ** 0.5),
        block_table=block_table,
        context_lens=context_lens,
        out=output
    )

    print("PageAttention 输出形状:", output.shape)