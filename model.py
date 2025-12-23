import torch.nn as nn
from encoder import TransformerEmbedding
from transformer import EncoderLayer


class SemanticMatchingModel(nn.Module):
    """
    语义匹配模型 (可返回注意力权重)
    """

    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=2, d_ff=512,
                 max_len=128, padding_idx=0, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            padding_idx=padding_idx,
            dropout=dropout
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.classifier = nn.Linear(d_model, 2)

    def forward(self, tokens, segment_ids, attention_mask, return_attentions=False):
        x = self.embedding(tokens, segment_ids)

        attention_mask_for_multihead = attention_mask.unsqueeze(1).unsqueeze(2)

        attentions = []
        for layer in self.encoder_layers:
            x = layer(x, attention_mask_for_multihead)
            if return_attentions:
                # 从 MultiHeadAttention 子模块中获取权重
                attentions.append(layer.self_attn.attention_weights)

        cls_token_output = x[:, 0, :]
        logits = self.classifier(cls_token_output)

        if return_attentions:
            return logits, attentions
        return logits