import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """输入编码 (Token Embedding)"""

    def __init__(self, vocab_size, d_model, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        return self.pos_embedding(position_ids)


class SegmentEmbedding(nn.Module):
    """分段编码"""

    def __init__(self, d_model, num_segments=2):
        super().__init__()
        self.segment_embedding = nn.Embedding(num_segments, d_model)

    def forward(self, segment_ids):
        return self.segment_embedding(segment_ids)


class TransformerEmbedding(nn.Module):
    """
    组装嵌入层
    将输入编码、分段编码和位置编码相加。
    """

    def __init__(self, vocab_size, d_model, max_len, padding_idx, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.position_embedding = PositionalEmbedding(d_model, max_len)
        self.segment_embedding = SegmentEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tokens, segment_ids):
        # tokens: (batch_size, seq_len)
        # segment_ids: (batch_size, seq_len)

        token_emb = self.token_embedding(tokens)
        pos_emb = self.position_embedding(tokens)  # Pass tokens just to get shape and device
        seg_emb = self.segment_embedding(segment_ids)

        # 三种嵌入相加
        combined_embedding = token_emb + pos_emb + seg_emb

        # Layer Norm 和 Dropout
        x = self.layer_norm(combined_embedding)
        x = self.dropout(x)

        return x


if __name__ == '__main__':
    from dataset import Vocab, AFQMCDataset, collate_fn, load_afqmc_data
    from torch.utils.data import DataLoader

    # (2) 打印嵌入层的输入输出进行验证
    print("\n--- 验证嵌入层 ---")

    # 准备数据
    train_file = './AFQMC/train.json'
    train_data = load_afqmc_data(train_file)
    train_tokens = [list(s1) + list(s2) for s1, s2, _ in train_data]
    reserved_tokens = ['<pad>', '<cls>', '<sep>', '<unk>']
    vocab = Vocab(train_tokens, min_freq=2, reserved_tokens=reserved_tokens)

    train_dataset = AFQMCDataset(train_data, vocab, max_len=64)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, collate_fn=collate_fn)

    token_ids, segment_ids, attention_mask, labels = next(iter(train_loader))

    # 初始化嵌入层
    d_model = 128
    embedding_layer = TransformerEmbedding(
        vocab_size=len(vocab),
        d_model=d_model,
        max_len=64,
        padding_idx=vocab['<pad>']
    )

    # 前向传播
    output = embedding_layer(token_ids, segment_ids)

    print("嵌入层输入 (Token IDs):", token_ids.shape)
    print("嵌入层输入 (Segment IDs):", segment_ids.shape)
    print("\n嵌入层输出 (shape):", output.shape)
    print("输出向量示例 (第一条数据的第一个token):\n", output[0, 0, :])