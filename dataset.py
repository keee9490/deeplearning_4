import torch
from torch.utils.data import Dataset, DataLoader
import json

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

def build_vocab(sentences, min_freq=1):
    # sentences: list of sentences, single string per sentence
    from collections import Counter
    char_count = Counter()
    for s in sentences:
        char_count.update(list(s))
    vocab = SPECIAL_TOKENS[:]
    for c, n in char_count.items():
        if n >= min_freq and c not in vocab:
            vocab.append(c)
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    return stoi, itos

def sent2ids(sent, stoi):
    # 字转ID
    return [stoi.get(c, stoi["[UNK]"]) for c in sent]

class AFQMCDataset(Dataset):
    def __init__(self, filepath, stoi, max_len=64):
        self.data = []
        with open(filepath, encoding='utf8') as f:
            for line in f:
                item = json.loads(line)
                s1 = item['sentence1']
                s2 = item['sentence2']
                lab = int(item['label'])
                self.data.append((s1, s2, lab))
        self.stoi = stoi
        self.max_len = max_len


    def __getitem__(self, idx):
        s1, s2, lab = self.data[idx]
        # 输入格式：[CLS] s1 [SEP] s2 [SEP]
        ids = [self.stoi["[CLS]"]] + sent2ids(s1, self.stoi) + [self.stoi["[SEP]"]]
        ids += sent2ids(s2, self.stoi) + [self.stoi["[SEP]"]]
        # 对齐&mask
        if len(ids) < self.max_len:
            ids = ids + [self.stoi["[PAD]"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        attn_mask = [int(i != self.stoi["[PAD]"]) for i in ids]
        return torch.tensor(ids), torch.tensor(attn_mask), torch.tensor(lab)

    def __len__(self):
        return len(self.data)

def afqmc_collate(batch):
    input_ids, attn_mask, labels = zip(*batch)
    return torch.stack(input_ids), torch.stack(attn_mask), torch.tensor(labels)

# 验证
if __name__ == "__main__":
    # 收集所有句构建词表
    data_tr = []
    with open('afqmc_train.txt', encoding='utf8') as f:
        for line in f: data_tr += line.strip().split('\t')[:2]
    data_val = []
    with open('afqmc_val.txt', encoding='utf8') as f:
        for line in f: data_val += line.strip().split('\t')[:2]
    stoi, itos = build_vocab(data_tr + data_val)

    ds = AFQMCDataset('afqmc_train.txt', stoi)
    dl = DataLoader(ds, batch_size=4, collate_fn=afqmc_collate, shuffle=True)
    for batch in dl:
        print("一个mini-batch的数据：")
        print('input_ids:', batch[0])
        print('attn_mask:', batch[1])
        print('labels:', batch[2])
        break