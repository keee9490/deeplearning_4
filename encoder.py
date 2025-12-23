import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=64, seg_num=2):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.seg_embed = nn.Embedding(seg_num, embed_dim)

    def forward(self, input_ids, seg_ids):
        device = input_ids.device
        pos_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand_as(input_ids)
        e = self.word_embed(input_ids) + self.pos_embed(pos_ids) + self.seg_embed(seg_ids)
        return e

if __name__ == "__main__":
    input_ids = torch.randint(0, 10, (2, 8))
    seg_ids = torch.randint(0, 2, (2, 8))
    layer = EmbeddingLayer(20, 32, max_len=8)
    out = layer(input_ids, seg_ids)
    print("嵌入层输入input_ids:\n", input_ids)
    print("分段seg_ids:\n", seg_ids)
    print("嵌入层输出shape:", out.shape)
    print("嵌入值举例:", out[0,0])