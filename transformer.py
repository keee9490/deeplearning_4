import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    多头自注意力层
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # 新增: 用于存储注意力权重
        self.attention_weights = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 保存softmax之前的权重
        self.attention_weights = torch.softmax(scores, dim=-1)

        attention_weights_with_dropout = self.dropout(self.attention_weights)

        context = torch.matmul(attention_weights_with_dropout, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(context)

        return output


class PositionWiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        _x = x
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(_x + self.dropout(attn_output))

        _x = x
        ff_output = self.feed_forward(x)
        x = self.norm2(_x + self.dropout(ff_output))

        return x