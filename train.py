import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import build_vocab, AFQMCDataset, afqmc_collate
from model import SemanticMatchModel

# 1. 载入词表与数据
all_sents = []
for fn in ['AFQMC/train.json', 'AFQMC/dev.json']:
    with open(fn, 'r', encoding='utf8') as f:
        for line in f: all_sents += line.strip().split('\t')[:2]
stoi, _ = build_vocab(all_sents)
batch_size, max_len = 32, 64

ds_tr = AFQMCDataset('AFQMC/train.json', stoi, max_len)
ds_val = AFQMCDataset('AFQMC/dev.json', stoi, max_len)
dl_tr = DataLoader(ds_tr, batch_size=batch_size, collate_fn=afqmc_collate, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=batch_size, collate_fn=afqmc_collate)

# 2. 建模
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SemanticMatchModel(len(stoi), d_model=128, max_len=max_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

writer = SummaryWriter("logs_afqmc")
best_acc, best_model_path = 0, 'model_best.pt'

def calc_acc(lbls, preds):
    preds = preds.argmax(1)
    return (preds == lbls).float().mean().item()

# 3. 训练循环
for epoch in range(10):
    model.train()
    loss_all, acc_all = 0, 0
    for i, (input_ids, attn_mask, labels) in enumerate(dl_tr):
        seg_ids = torch.zeros_like(input_ids)
        for j in range(input_ids.shape[0]):
            # [CLS]句1[SEP]句2[SEP], 构造分段：0是句1和[CLS][SEP]，1是句2（和后面第2个[SEP]）
            sep_idx = (input_ids[j] == stoi["[SEP]"]).nonzero().flatten()
            if len(sep_idx) >= 1:
                seg_ids[j, (sep_idx[0]+1):] = 1
        input_ids, seg_ids, attn_mask, labels = [t.to(device) for t in [input_ids, seg_ids, attn_mask, labels]]
        logits = model(input_ids, seg_ids, attn_mask)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = calc_acc(labels, logits)
        loss_all += loss.item()
        acc_all += acc
        writer.add_scalar('train/loss', loss.item(), epoch * len(dl_tr) + i)
        writer.add_scalar('train/acc', acc, epoch * len(dl_tr) + i)

    # 验证
    model.eval()
    val_loss, val_acc, val_cnt = 0, 0, 0
    with torch.no_grad():
        for input_ids, attn_mask, labels in dl_val:
            seg_ids = torch.zeros_like(input_ids)
            for j in range(input_ids.shape[0]):
                sep_idx = (input_ids[j] == stoi["[SEP]"]).nonzero().flatten()
                if len(sep_idx) >= 1:
                    seg_ids[j, (sep_idx[0]+1):] = 1
            input_ids, seg_ids, attn_mask, labels = [t.to(device) for t in [input_ids, seg_ids, attn_mask, labels]]
            logits = model(input_ids, seg_ids, attn_mask)
            loss = criterion(logits, labels)
            acc = calc_acc(labels, logits)
            val_loss += loss.item() * len(labels)
            val_acc += acc * len(labels)
            val_cnt += len(labels)

    avg_acc = val_acc / val_cnt
    writer.add_scalar('val/acc', avg_acc, epoch)
    writer.add_scalar('val/loss', val_loss / val_cnt, epoch)
    print(f"Epoch {epoch}: Val Acc={avg_acc:.3f} val_loss={val_loss/val_cnt:.3f}")
    if avg_acc > best_acc:
        torch.save(model.state_dict(), best_model_path)
        best_acc = avg_acc
        print("模型已保存:", best_model_path)

print("最终验证准确率：", best_acc)
writer.close()