import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from dataset import Vocab, AFQMCDataset, collate_fn, load_afqmc_data
from model import SemanticMatchingModel


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch + 1}")
    for i, (tokens, segments, mask, labels) in enumerate(progress_bar):
        tokens, segments, mask, labels = tokens.to(device), segments.to(device), mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(tokens, segments, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}', 'acc': f'{total_correct / total_samples:.4f}'})

        # 记录到TensorBoard
        step = epoch * len(dataloader) + i
        writer.add_scalar('Loss/train', loss.item(), step)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    writer.add_scalar('Accuracy/train_epoch', accuracy, epoch)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for tokens, segments, mask, labels in tqdm(dataloader, desc=f"Eval Epoch {epoch + 1}"):
            tokens, segments, mask, labels = tokens.to(device), segments.to(device), mask.to(device), labels.to(device)
            outputs = model(tokens, segments, mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    writer.add_scalar('Loss/validation', avg_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    return avg_loss, accuracy


def main():
    # 超参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    MAX_LEN = 64
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1

    # 路径
    train_file = './AFQMC/train.json'
    dev_file = './AFQMC/dev.json'
    vocab_path = './AFQMC/vocab.pth'
    model_save_path = './AFQMC/best_model.pth'
    log_dir = './logs_afqmc'

    # 1. 准备数据和词表
    print("Preparing data and vocabulary...")
    train_data = load_afqmc_data(train_file)
    dev_data = load_afqmc_data(dev_file)

    train_tokens = [list(s1) + list(s2) for s1, s2, _ in train_data]
    reserved_tokens = ['<pad>', '<cls>', '<sep>', '<unk>']
    vocab = Vocab(train_tokens, min_freq=2, reserved_tokens=reserved_tokens)
    torch.save(vocab, vocab_path)
    print(f"Vocabulary size: {len(vocab)}, saved to {vocab_path}")

    train_dataset = AFQMCDataset(train_data, vocab, max_len=MAX_LEN)
    dev_dataset = AFQMCDataset(dev_data, vocab, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 2. 初始化模型、损失函数和优化器
    model = SemanticMatchingModel(
        vocab_size=len(vocab), d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_len=MAX_LEN, padding_idx=vocab['<pad>'], dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 初始化TensorBoard
    writer = SummaryWriter(log_dir)

    # 4. 训练和评估循环
    best_val_accuracy = 0.0
    print(f"\nStart training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, writer)
        val_loss, val_acc = evaluate(model, dev_loader, criterion, DEVICE, epoch, writer)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 5. 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"--- Best model saved with validation accuracy: {best_val_accuracy:.4f} ---")

    writer.close()
    print(f"\nTraining finished. Best model saved at {model_save_path}")


if __name__ == "__main__":
    main()