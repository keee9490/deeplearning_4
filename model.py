import torch
import torch.nn as nn
from encoder import EmbeddingLayer
from transformer import MultiHeadSelfAttention, AddNorm

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.addnorm1 = AddNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.addnorm2 = AddNorm(d_model)

    def forward(self, x, mask):
        h = self.addnorm1(x, self.attn(x, mask))
        h2 = self.addnorm2(h, self.ffn(h))
        return h2

class SemanticMatchModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=64, seg_num=2, n_layers=2, n_heads=4, d_ff=256, n_class=2):
        super().__init__()
        self.embed = EmbeddingLayer(vocab_size, d_model, max_len, seg_num)
        self.encoders = nn.ModuleList([TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, n_class)
        )

    def forward(self, input_ids, seg_ids, mask):
        x = self.embed(input_ids, seg_ids)
        for encoder in self.encoders:
            x = encoder(x, mask)
        cls_vec = x[:, 0, :]  # 取[CLS]位置
        logits = self.classifier(cls_vec)
        return logits