import torch
import random
import os

from model import SemanticMatchingModel
from dataset import Vocab, AFQMCDataset, load_afqmc_data


def test_model():
    # 超参数 (需要和训练时保持一致)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LEN = 64
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1

    # 路径
    dev_file = './AFQMC/dev.json'
    vocab_path = './AFQMC/vocab.pth'
    model_path = './AFQMC/best_model.pth'

    # 1. 加载词表和模型
    print("Loading vocabulary and model...")
    if not all(os.path.exists(p) for p in [vocab_path, model_path, dev_file]):
        print("Error: vocab.pth, best_model.pth or dev.json not found.")
        print("Please run train.py first to create these files.")
        return

    vocab = torch.load(vocab_path)
    model = SemanticMatchingModel(
        vocab_size=len(vocab), d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_len=MAX_LEN, padding_idx=vocab['<pad>'], dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 准备数据
    dev_data = load_afqmc_data(dev_file)

    # 3. 随机选取50条数据进行测试
    num_samples = 50
    sample_data = random.sample(dev_data, num_samples)

    print(f"\n--- Running prediction on {num_samples} random samples from validation set ---\n")

    correct_predictions = 0

    for s1, s2, label in sample_data:
        # 手动进行与Dataset中相同的预处理
        s1_tokens = vocab[list(s1)]
        s2_tokens = vocab[list(s2)]

        tokens = [vocab['<cls>']] + s1_tokens + [vocab['<sep>']] + s2_tokens
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]

        segment_ids = [0] * (len(s1_tokens) + 2) + [1] * (len(s2_tokens))
        if len(segment_ids) > MAX_LEN:
            segment_ids = segment_ids[:MAX_LEN]

        # 转换为tensor并增加batch维度
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
        segments_tensor = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        attention_mask = (tokens_tensor != vocab['<pad>']).long().to(DEVICE)

        with torch.no_grad():
            output = model(tokens_tensor, segments_tensor, attention_mask)
            _, predicted = torch.max(output.data, 1)

        prediction_text = "相似" if predicted.item() == 1 else "不相似"
        label_text = "相似" if label == 1 else "不相似"

        if predicted.item() == label:
            correct_predictions += 1
            result = "✓ (正确)"
        else:
            result = "✗ (错误)"

        print(f"句子1: {s1}")
        print(f"句子2: {s2}")
        print(f"真实标签: {label_text} | 模型预测: {prediction_text} -> {result}\n")

    print(f"--- Test Finished ---")
    print(f"Accuracy on {num_samples} samples: {correct_predictions / num_samples * 100:.2f}%")


if __name__ == "__main__":
    test_model()