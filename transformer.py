import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model*3)
        self.fc_out = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B, T, d_model = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, T, d
        att = (q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)
        if mask is not None:
            # mask为(B, T)，转(B,1,T)，再广播到(B,nH,T,T)
            mask = mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(mask==0, -1e9)
        att = self.softmax(att)
        out = att @ v
        out = out.transpose(1,2).contiguous().reshape(B,T,d_model)
        return self.fc_out(out)

class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_out):
        return self.norm(x + sublayer_out)

# Test
if __name__ == "__main__":
    x = torch.randn(2, 8, 32)
    mask = torch.ones(2,8)
    attn = MultiHeadSelfAttention(32, 4)
    out = attn(x, mask)
    print("多头注意力输出shape", out.shape)
    # Add&Norm
    addnorm = AddNorm(32)
    result = addnorm(x, out)
    print("Add&Norm输出shape", result.shape)