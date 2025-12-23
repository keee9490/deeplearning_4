import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

from model import SemanticMatchingModel
from dataset import Vocab


def visualize_attention(sentence1, sentence2, model, vocab, device, max_len=64):
    """
    对单个样本进行预测并可视化其所有头的注意力权重。
    """
    model.eval()

    # 1. 预处理输入句子
    s1_tokens_char = list(sentence1)
    s2_tokens_char = list(sentence2)
    s1_tokens_id = vocab[s1_tokens_char]
    s2_tokens_id = vocab[s2_tokens_char]

    tokens_id = [vocab['<cls>']] + s1_tokens_id + [vocab['<sep>']] + s2_tokens_id
    if len(tokens_id) > max_len:
        tokens_id = tokens_id[:max_len]

    segment_ids = [0] * (len(s1_tokens_id) + 2) + [1] * (len(s2_tokens_id))
    if len(segment_ids) > max_len:
        segment_ids = segment_ids[:max_len]

    # Padding
    padding_len = max_len - len(tokens_id)
    tokens_id += [vocab['<pad>']] * padding_len
    segment_ids += [0] * padding_len

    # 转换为Tensor
    tokens_tensor = torch.tensor(tokens_id, dtype=torch.long).unsqueeze(0).to(device)
    segments_tensor = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = (tokens_tensor != vocab['<pad>']).long().to(device)

    # 2. 获取模型输出和注意力权重
    with torch.no_grad():
        _, attentions = model(tokens_tensor, segments_tensor, attention_mask, return_attentions=True)

    # 3. 准备可视化
    #   - `attentions` 是一个列表, 包含模型每一层的注意力权重
    #   - 我们只可视化最后一层的注意力
    #   - 形状: (batch_size, num_heads, seq_len, seq_len)
    last_layer_attention = attentions[-1][0]  # (num_heads, seq_len, seq_len)

    # 将padding部分去掉，以便可视化
    actual_seq_len = sum(attention_mask[0]).item()
    last_layer_attention = last_layer_attention[:, :actual_seq_len, :actual_seq_len]

    # 获取真实的tokens
    tokens_char = ['<cls>'] + s1_tokens_char + ['<sep>'] + s2_tokens_char
    if len(tokens_char) > max_len:
        tokens_char = tokens_char[:max_len]

    num_heads = last_layer_attention.size(0)

    # 中文显示设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 为每个头绘制一个热力图
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(20, 8))
    axes = axes.ravel()

    for i in range(num_heads):
        ax = axes[i]
        sns.heatmap(last_layer_attention[i].cpu().numpy(), ax=ax, cmap='Reds',
                    xticklabels=tokens_char, yticklabels=tokens_char)
        ax.set_title(f'Attention Head {i + 1}')
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('Last Layer Attention Weights', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("attention_visualization.png")
    print("注意力权重可视化图像已保存为 attention_visualization.png")
    plt.show()


if __name__ == '__main__':
    # --- 参数设置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LEN = 64
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1

    vocab_path = './AFQMC/vocab.pth'
    model_path = './AFQMC/best_model.pth'

    # --- 样本句子 ---
    # 这个样本与您图片中的类似
    sentence1 = "电脑怎么录像？"
    sentence2 = "如何在计算机上录视频"

    # --- 加载模型和词汇表 ---
    print("Loading vocabulary and model...")
    if not all(os.path.exists(p) for p in [vocab_path, model_path]):
        print("Error: vocab.pth or best_model.pth not found.")
        print("Please run train.py first.")
    else:
        vocab = torch.load(vocab_path)
        model = SemanticMatchingModel(
            vocab_size=len(vocab), d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            d_ff=D_FF, max_len=MAX_LEN, padding_idx=vocab['<pad>'], dropout=DROPOUT
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # --- 可视化注意力 ---
        visualize_attention(sentence1, sentence2, model, vocab, DEVICE, max_len=MAX_LEN)