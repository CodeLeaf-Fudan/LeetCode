import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        assert self.head_dim * num_heads == embed_dim # 检查embed_dim能被num_heads整除

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)  # QKV线性投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)      # 输出线性投影

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # 将输入投影为Q、K、V，并调整形状
        qkv = self.qkv_proj(x).view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别为Q、K、V，形状为(batch_size, num_heads, seq_length, head_dim)

        # 计算缩放点积注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = torch.softmax(scores, dim=-1)  # 对最后一个维度做softmax

        # 用注意力权重加权V
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length, head_dim)
        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        return self.out_proj(attn_output)  # 输出线性投影

# Example usage
if __name__ == "__main__":
    embed_dim = 64
    num_heads = 8
    batch_size = 2
    seq_length = 10

    model = MultiHeadAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_length, embed_dim)  # 随机输入

    output = model(x)
    print("Output shape:", output.shape)  # 输出形状应为(batch_size, seq_length, embed_dim)