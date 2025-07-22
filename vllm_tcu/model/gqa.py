import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttentionDirect(nn.Module):
    def __init__(
            self,
            num_heads: int,  # 查询头总数（如32）
            num_groups: int,  # 分组数量（如8）
            head_dim: int  # 单头维度（如128）
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = head_dim

        # 确保头数能被组数整除
        if num_heads % num_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups})"
            )

        # 每组的查询头数
        self.heads_per_group = num_heads // num_groups

        # 输出投影层（可选）
        self.out_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim,dtype=torch.bfloat16)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None,
            scale_value: float = None
    ) -> torch.Tensor:
        """
        接受预先拆分头维度的输入：
        query: [seq_len, num_heads, head_dim]
        key:   [seq_len, num_groups, head_dim]
        value: [seq_len, num_groups, head_dim]
        mask:  [query_seq, key_seq]
        """
        print(f"ATTENTION :gqa shapes:query:{query.shape},key:{key.shape},value:{value.shape},"
              f"mask:{mask.shape},scale_value:{scale_value}");
        # 1. 处理维度
        # 输入形状处理：[seq_len, ...] -> [1, seq_len, ...] 以支持批量维度
        q = query.unsqueeze(0) if query.dim() == 3 else query  # [1, seq_len, num_heads, head_dim]
        k = key.unsqueeze(0) if key.dim() == 3 else key  # [1, seq_len, num_groups, head_dim]
        v = value.unsqueeze(0) if value.dim() == 3 else value  # [1, seq_len, num_groups, head_dim]

        # 2. 维度转换
        # query: [1, seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        # key/value: [1, seq_len, num_groups, head_dim] -> [1, num_groups, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 3. 分组处理
        # [1, num_heads, seq_len, head_dim] -> [1, num_groups, heads_per_group, seq_len, head_dim]
        q = q.view(1, self.num_groups, self.heads_per_group, q.size(2), self.head_dim)

        # 4. 注意力计算
        # [batch, groups, heads, q_len, head_dim] @ [batch, groups, head_dim, k_len]
        # -> [batch, groups, heads, q_len, k_len]
        attn_scores = torch.einsum('bghqd,bgkd->bghqk', q, k)

        # 5. 应用缩放因子（默认使用head_dim的倒数平方根）
        scale = scale_value if scale_value is not None else self.head_dim ** -0.5
        attn_scores = attn_scores * scale

        # 6. 应用注意力掩码
        if mask is not None:
            # 处理mask维度：[query_seq, key_seq] -> [1, 1, 1, q_seq, k_seq]
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # 将负无穷位置设为极小值  comment it triggers mismatch , go through first .
            # attn_scores = attn_scores + mask.masked_fill(mask == 0, -1e10)

        # 7. Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 8. 加权求和
        # [batch, groups, heads, q_len, k_len] @ [batch, groups, k_len, head_dim]
        # -> [batch, groups, heads, q_len, head_dim]
        output = torch.einsum('bghqk,bgkd->bghqd', attn_weights, v)

        # 9. 合并组结果
        # 合并头维度: [1, groups, heads, seq_len, head_dim] -> [1, heads, seq_len, head_dim]
        output = output.view(1, self.num_heads, output.size(3), self.head_dim)

        # 10. 恢复原始维度顺序: [1, heads, seq_len, head_dim] -> [1, seq_len, heads, head_dim]
        output = output.permute(0, 2, 1, 3)

        # 11. 输出投影（可选）
        if self.out_proj:
            orig_shape = output.shape
            output = self.out_proj(output.reshape(orig_shape[0], orig_shape[1], -1))
            output = output.view(orig_shape)

        # 12. 移除批量维度: [1, seq_len, heads, head_dim] -> [seq_len, heads, head_dim]
        return output.squeeze(0)


# 使用示例
if __name__ == "__main__":
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 参数设置
    seq_len = 51
    num_heads = 32
    num_groups = 8
    head_dim = 128
    scale = 0.08838834764831845

    # 创建输入
    query = torch.randn([seq_len, num_heads, head_dim], device=device, dtype=torch.bfloat16)
    key = torch.randn([seq_len, num_groups, head_dim], device=device, dtype=torch.bfloat16)
    value = torch.randn([seq_len, num_groups, head_dim], device=device, dtype=torch.bfloat16)
    mask = torch.randn([seq_len, seq_len], device=device, dtype=torch.bfloat16)

    # 初始化GQA层
    gqa = GroupedQueryAttentionDirect(num_heads, num_groups, head_dim)

    # 转移模型到设备
    gqa.to(device)

    # 调用注意力层
    output = gqa(
        query=query,
        key=key,
        value=value,
        mask=mask,
        scale_value=scale
    )

    print(f"输入 query 形状: {query.shape}")
    print(f"输入 key 形状: {key.shape}")
    print(f"输入 value 形状: {value.shape}")
    print(f"输出形状: {output.shape}")  # [51, 32, 128]