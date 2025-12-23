import torch
import json
from torch.utils.data import Dataset, DataLoader


class Vocab:
    """词汇表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = self._count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元和预留词元的索引
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def _count_corpus(self, tokens):
        # 这里的tokens是1D列表或2D列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for line in tokens for token in line]
        from collections import Counter
        return Counter(tokens)


def load_afqmc_data(file_path):
    """加载AFQMC数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line.strip())
            data.append((row['sentence1'], row['sentence2'], int(row['label'])))
    return data


class AFQMCDataset(Dataset):
    """AFQMC数据集的Dataset实现"""

    def __init__(self, data, vocab, max_len=128):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s1, s2, label = self.data[idx]

        # 1. 将句子中的中文字符转换为id
        s1_tokens = self.vocab[list(s1)]
        s2_tokens = self.vocab[list(s2)]

        # 2. 在起始位置加入占位符[CLS]，在输入中加入句子的分隔符号[SEP]
        # 格式: [CLS] s1 [SEP] s2
        # 对不在词汇表里面的字做出适当处理 (Vocab类已处理，会映射到<unk>)
        cls_id = self.vocab['<cls>']
        sep_id = self.vocab['<sep>']

        tokens = [cls_id] + s1_tokens + [sep_id] + s2_tokens

        # 分段编码
        segment_ids = [0] * (len(s1_tokens) + 2) + [1] * (len(s2_tokens))

        # 截断
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            segment_ids = segment_ids[:self.max_len]

        return tokens, segment_ids, label


def collate_fn(batch):
    """
    小批量数据的组装及对齐
    batch: 一个列表，列表的每个元素是Dataset的__getitem__返回的结果
    """
    tokens, segment_ids, labels = zip(*batch)

    max_len = max(len(t) for t in tokens)

    # 对齐 (padding)
    padded_tokens = []
    attention_masks = []
    padded_segment_ids = []
    pad_id = 0  # <pad> 的 id

    for i in range(len(tokens)):
        cur_len = len(tokens[i])

        # Padding
        padded_tok = tokens[i] + [pad_id] * (max_len - cur_len)
        padded_seg = segment_ids[i] + [0] * (max_len - cur_len)  # segment_id for pad can be 0 or any other value

        # Attention Mask
        attn_mask = [1] * cur_len + [0] * (max_len - cur_len)

        padded_tokens.append(padded_tok)
        attention_masks.append(attn_mask)
        padded_segment_ids.append(padded_seg)

    return (
        torch.tensor(padded_tokens, dtype=torch.long),
        torch.tensor(padded_segment_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )


if __name__ == '__main__':
    # 假设数据集在 'AFQMC' 文件夹下
    train_file = './AFQMC/train.json'
    dev_file = './AFQMC/dev.json'

    # 加载数据
    train_data = load_afqmc_data(train_file)
    dev_data = load_afqmc_data(dev_file)

    # 从训练数据构建词汇表
    train_tokens = []
    for s1, s2, _ in train_data:
        train_tokens.append(list(s1))
        train_tokens.append(list(s2))

    # 预留特殊词元
    reserved_tokens = ['<pad>', '<cls>', '<sep>', '<unk>']
    vocab = Vocab(train_tokens, min_freq=2, reserved_tokens=reserved_tokens)
    print(f"词汇表大小: {len(vocab)}")

    # 实例化Dataset和DataLoader
    batch_size = 4
    train_dataset = AFQMCDataset(train_data, vocab, max_len=64)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 验证: 打印一条mini-batch的数据
    print("\n--- 验证一个 Mini-Batch 的数据 ---")
    for i, batch in enumerate(train_dataloader):
        # batch 包含: token_ids, segment_ids, attention_mask, labels
        token_ids, segment_ids, attention_mask, labels = batch

        print(f"Batch {i + 1}:")
        print(f"Token IDs (shape): {token_ids.shape}")
        print(f"Segment IDs (shape): {segment_ids.shape}")
        print(f"Attention Mask (shape): {attention_mask.shape}")
        print(f"Labels (shape): {labels.shape}")

        print("\n第一条数据的详情:")
        print(f"Token IDs: {token_ids[0]}")
        print(f"对应的Tokens: {vocab.to_tokens(token_ids[0].tolist())}")
        print(f"Segment IDs: {segment_ids[0]}")
        print(f"Attention Mask: {attention_mask[0]}")
        print(f"Label: {labels[0]}")

        break  # 只打印第一个batch