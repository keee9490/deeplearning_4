import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

# 确保可以从其他文件中导入
from dataset import Vocab, AFQMCDataset, collate_fn, load_afqmc_data
from model import SemanticMatchingModel


def run_single_experiment(n_layers, device, dataloaders, vocab):
    """
    为指定的层数完整地运行一次训练和评估。
    """
    # --- 超参数 ---
    NUM_EPOCHS = 10  # 为了快速实验，可以适当减少Epochs
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    MAX_LEN = 64
    D_MODEL = 128
    N_HEADS = 8
    D_FF = 512
    DROPOUT = 0.1

    # --- 路径设置 ---
    model_save_path = f'./AFQMC/best_model_{n_layers}_layers.pth'

    print(f"\n{'=' * 20} 正在开始: {n_layers} 层 Transformer 实验 {'=' * 20}")
    start_time = time.time()

    # 1. 初始化模型
    model = SemanticMatchingModel(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=n_layers,  # 使用传入的层数
        d_ff=D_FF,
        max_len=MAX_LEN,
        padding_idx=vocab['<pad>'],
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    best_val_accuracy = 0.0

    # 2. 训练循环
    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        for tokens, segments, mask, labels in tqdm(train_loader,
                                                   desc=f"Train Epoch {epoch + 1}/{NUM_EPOCHS} (L={n_layers})"):
            tokens, segments, mask, labels = tokens.to(device), segments.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tokens, segments, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for tokens, segments, mask, labels in dev_loader:
                tokens, segments, mask, labels = tokens.to(device), segments.to(device), mask.to(device), labels.to(
                    device)
                outputs = model(tokens, segments, mask)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        val_acc = total_correct / total_samples
        print(f"Epoch {epoch + 1} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)

    end_time = time.time()
    print(f"--- {n_layers} 层实验完成, 耗时: {end_time - start_time:.2f} 秒 ---")
    print(f"--- 最佳验证集准确率: {best_val_accuracy:.4f} (模型保存在: {model_save_path}) ---")

    return best_val_accuracy


if __name__ == '__main__':
    # --- 实验配置 ---
    # 定义要测试的不同Transformer层数
    LAYER_CONFIGS = [1, 2, 4, 6]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 数据准备 (只做一次) ---
    print(">>> 正在准备数据集和词汇表...")
    train_file = './AFQMC/train.json'
    dev_file = './AFQMC/dev.json'

    if not (os.path.exists(train_file) and os.path.exists(dev_file)):
        print("错误: 找不到 train.json 或 dev.json 文件。")
    else:
        train_data = load_afqmc_data(train_file)
        dev_data = load_afqmc_data(dev_file)

        train_tokens = [list(s1) + list(s2) for s1, s2, _ in train_data]
        reserved_tokens = ['<pad>', '<cls>', '<sep>', '<unk>']
        vocab = Vocab(train_tokens, min_freq=2, reserved_tokens=reserved_tokens)

        train_dataset = AFQMCDataset(train_data, vocab, max_len=64)
        dev_dataset = AFQMCDataset(dev_data, vocab, max_len=64)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn),
            'dev': DataLoader(dev_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        }
        print(">>> 数据准备完成。")

        # --- 运行所有实验 ---
        results = {}
        for n in LAYER_CONFIGS:
            accuracy = run_single_experiment(n_layers=n, device=DEVICE, dataloaders=dataloaders, vocab=vocab)
            results[n] = accuracy

        # --- 汇总并打印最终结果 ---
        print("\n\n==================== 实验总结 ====================")
        print("Transformer层数对模型在验证集上准确率的影响:")
        print("--------------------------------------------------")
        print("|   层数 (N_LAYERS)   |   最佳验证集准确率   |")
        print("--------------------------------------------------")
        for layers, acc in results.items():
            print(f"|{layers:^21}|{acc:^24.4f}|")
        print("--------------------------------------------------")
