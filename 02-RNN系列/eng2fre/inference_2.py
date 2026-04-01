import numpy as np
import torch
from matplotlib import pyplot as plt

from model import EncoderRNN, AttentionDecoderRNN, MAX_LENGTH
from data_processing import device, normalizeString, data_loader, EOS_token, SOS_token


# 【修改】：增加 actual_input_len 参数，用于截取有效注意力列
def seq2seq_eval(x, encoderRnn, atten_decoderRnn, fre_idx2word, actual_input_len):
    h0 = encoderRnn.initHidden(1)
    encoder_output, encode_hidden = encoderRnn(x, h0)

    encoder_outputs_c = encoder_output
    decoder_hidden = encode_hidden
    decoder_input = torch.tensor([[SOS_token]], device=device, dtype=torch.long)

    seq_list = []
    # 【修改】：使用列表动态收集每一时刻的注意力权重
    full_attn_list = []

    for idx in range(MAX_LENGTH):
        output, hidden, attn_weights = atten_decoderRnn(decoder_input, decoder_hidden, encoder_outputs_c)

        # 【新增】：保存当前步骤的注意力分布 [1, 1, 12] -> [12]
        full_attn_list.append(attn_weights.view(-1).detach().cpu().numpy())

        input_y = torch.argmax(output, dim=-1).view(1, -1)
        decoder_input = input_y

        if input_y.item() == EOS_token:
            seq_list.append('<EOS>')
            break
        else:
            seq_list.append(fre_idx2word[input_y.item()])

    # 【核心修改】：将列表转为矩阵 [输出长度, 12]
    atten_matrix = np.array(full_attn_list)

    # 【核心修改】：根据实际输入长度进行切片，只保留有效单词和EOS的列
    trimmed_atten = atten_matrix[:, :actual_input_len]

    return seq_list, trimmed_atten


def plot_attention():
    dataloader, eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num = data_loader()

    encoderRnn = EncoderRNN(eng_vocab_size=eng_word_num, hidden_size=32).to(device)
    atten_decoderRnn = AttentionDecoderRNN(fre_vocab_size=fre_word_num, hidden_size=32).to(device)

    encoderRnn.load_state_dict(torch.load(r'./model/encoderRnn.pth', map_location=device))
    atten_decoderRnn.load_state_dict(torch.load(r'./model/atten_decoderRnn.pth', map_location=device))

    encoderRnn.eval()
    atten_decoderRnn.eval()

    # 原始数据
    sentence = 'she is beautiful like her mother .'
    normalized_sent = normalizeString(sentence)
    input_words = normalized_sent.split(' ')

    # 【新增】：计算实际输入长度（单词数 + 1个EOS）
    # 这将作为热力图的横坐标宽度
    actual_input_len = len(input_words) + 1

    # 数据处理逻辑保持不变，确保模型输入长度为 12 (MAX_LENGTH + 1)
    x = [eng_word2idx[word] for word in input_words]
    if len(x) < MAX_LENGTH:
        x += [2] * (MAX_LENGTH - len(x))
    else:
        x = x[:MAX_LENGTH]
    x.append(EOS_token)

    x_tensor = torch.tensor(x, dtype=torch.long).unsqueeze(0).to(device)

    # 【修改】：传入 actual_input_len 得到切片后的注意力阵
    fre_word_list, atten = seq2seq_eval(x_tensor, encoderRnn, atten_decoderRnn, fre_idx2word, actual_input_len)

    # --- 绘图部分修改 ---
    fig, ax = plt.subplots(figsize=(10, 8))  # 稍微调大画布
    cax = ax.matshow(atten, cmap='viridis')
    fig.colorbar(cax)

    # 【新增】：设置坐标轴标签，使热力图具备可读性
    # 横坐标是输入英文
    ax.set_xticklabels([''] + input_words + ['<EOS>'], rotation=45)
    # 纵坐标是输出法文
    ax.set_yticklabels([''] + fre_word_list)

    # 自动设置坐标刻度
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    # 填充数值
    for (i, j), value in np.ndenumerate(atten):
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white')

    plt.savefig("s2s_attn_fixed.png")
    plt.show()


if __name__ == '__main__':
    plot_attention()