import numpy as np
import torch
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt

# 强制使用 TkAgg 后端以支持窗口显示（根据你的配置保留）
# matplotlib.use('TkAgg')
#
# # 设置中文字体（确保你的系统支持这些字体，否则热力图标签可能乱码）
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False

from model import EncoderRNN, AttentionDecoderRNN, MAX_LENGTH
from data_processing import device, normalizeString, data_loader, EOS_token, SOS_token, PAD_token


def seq2seq_eval(x_tensor, encoderRnn, atten_decoderRnn, fre_idx2word):
    """
    模型推理函数
    """
    encoderRnn.eval()
    atten_decoderRnn.eval()

    with torch.no_grad():
        # 1. Encoder 编码
        h0 = encoderRnn.initHidden(1)
        encoder_output, encode_hidden = encoderRnn(x_tensor, h0)

        decoder_hidden = encode_hidden
        # 构建初始输入 [SOS]
        decoder_input = torch.tensor([[SOS_token]], device=device, dtype=torch.long)

        seq_list = []
        # 注意力矩阵：行代表生成的法语单词，列代表输入的英语单词
        # 这里的列数必须与 encoder_output 的序列长度一致，即 MAX_LENGTH + 1 (包含 EOS)
        atten_matrix = torch.zeros(MAX_LENGTH + 1, MAX_LENGTH + 1, device=device)

        for idx in range(MAX_LENGTH + 1):
            output, decoder_hidden, attn_weights = atten_decoderRnn(decoder_input, decoder_hidden, encoder_output)

            # 存储注意力权重 [1, 1, MAX_LENGTH+1] -> [MAX_LENGTH+1]
            atten_matrix[idx] = attn_weights.squeeze()

            # 取概率最大的单词
            topv, topi = output.topk(1)
            input_y = topi.squeeze().detach()

            if input_y.item() == EOS_token:
                seq_list.append('<EOS>')
                break
            else:
                seq_list.append(fre_idx2word[input_y.item()])

            # 下一步的输入是当前的预测输出
            decoder_input = topi.view(1, -1)

    return seq_list, atten_matrix[:len(seq_list)]


def plot_attention():
    # 1. 加载词表和环境
    dataloader, eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num = data_loader()

    # 2. 模型实例化与加载
    encoderRnn = EncoderRNN(eng_vocab_size=eng_word_num, hidden_size=128).to(device)
    # 注意：max_length 必须与训练时的 MAX_LENGTH + 1 保持严格一致
    atten_decoderRnn = AttentionDecoderRNN(fre_vocab_size=fre_word_num, hidden_size=128, max_length=MAX_LENGTH + 1).to(
        device)

    encoderRnn.load_state_dict(torch.load(r'./model/encoderRnn.pth', map_location=device))
    atten_decoderRnn.load_state_dict(torch.load(r'./model/atten_decoderRnn.pth', map_location=device))

    # 3. 准备测试数据
    raw_x = 'she is beautiful like her mother .'
    normalized_x = normalizeString(raw_x)
    input_words = normalized_x.split(' ')

    # 转换为索引并处理长度（逻辑需与 data_processing.py 严格一致）
    x_indices = [eng_word2idx[word] for word in input_words]
    if len(x_indices) < MAX_LENGTH:
        x_indices += [PAD_token] * (MAX_LENGTH - len(x_indices))
    else:
        x_indices = x_indices[:MAX_LENGTH]

    # 最终输入序列包含一个 EOS
    full_x_indices = x_indices + [EOS_token]
    x_tensor = torch.tensor(full_x_indices, dtype=torch.long, device=device).unsqueeze(0)

    # 4. 推理
    fre_word_list, atten = seq2seq_eval(x_tensor, encoderRnn, atten_decoderRnn, fre_idx2word)

    # 5. 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # 转换为 numpy 绘图
    # 截取有效长度：只显示实际输入的单词（包含 PAD/EOS）和 生成的法语单词
    display_atten = atten.cpu().detach().numpy()

    cax = ax.matshow(display_atten, cmap='viridis')
    fig.colorbar(cax)

    # 设置坐标轴标签
    # X轴是输入的英语（源语言）
    # 这里的标签需要对应 full_x_indices
    x_labels = [eng_idx2word[idx.item()] for idx in x_tensor[0]]
    ax.set_xticklabels([''] + x_labels, rotation=45)

    # Y轴是输出的法语（目标语言）
    ax.set_yticklabels([''] + fre_word_list)

    # 设置刻度在每个单元格中心
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # 在格子内添加数值（可选，如果格子太小建议注释掉）
    for i in range(display_atten.shape[0]):
        for j in range(display_atten.shape[1]):
            ax.text(j, i, f'{display_atten[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

    plt.xlabel('Source Language (English)')
    plt.ylabel('Target Language (French)')
    plt.title('Attention Heatmap (Seq2Seq)')

    plt.tight_layout()
    plt.savefig("s2s_attn_improved.png")
    plt.show()


def eval():
    # 模型 训练好的模型
    # 1. 加载词表和环境
    dataloader, eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num = data_loader()

    # 2. 模型实例化与加载
    encoderRnn = EncoderRNN(eng_vocab_size=eng_word_num, hidden_size=128).to(device)
    # 注意：max_length 必须与训练时的 MAX_LENGTH + 1 保持严格一致
    atten_decoderRnn = AttentionDecoderRNN(fre_vocab_size=fre_word_num, hidden_size=128, max_length=MAX_LENGTH + 1).to(
        device)

    encoderRnn.load_state_dict(torch.load(r'./model/encoderRnn.pth', map_location=device))
    atten_decoderRnn.load_state_dict(torch.load(r'./model/atten_decoderRnn.pth', map_location=device))
    # 设置为eval模式
    encoderRnn.eval()
    atten_decoderRnn.eval()

    # 数据
    samplepairs = [['I m impressed with your french .', 'je suis impressionne par votre francais .'],
                   ['i m more than a friend .', 'je suis plus qu une amie .'],
                   ['she is beautiful like her mother .', 'elle est belle comme sa mere .']]
    # 数据处理
    for pair in samplepairs:
        # 获取英文和法文
        x = pair[0]
        y = pair[1]
        # 清洗
        x = normalizeString(x)
        print(x)
        # 英文分词 id
        word_list = [eng_word2idx[word] for word in x.split(' ')]
        # 添加截止符
        word_list.append(EOS_token)
        # 转化成tensor
        wordids = torch.tensor(word_list, dtype=torch.long).unsqueeze(0).to(device)
        # 模型推理
        fre_word_list, atten = seq2seq_eval(wordids, encoderRnn, atten_decoderRnn, fre_idx2word)

        # 拼接词语 获取结果
        frech_result = ' '.join(fre_word_list)
        print(atten)
        print('预测结果：', frech_result, '真实结果：', y)
    return frech_result


if __name__ == '__main__':
    # plot_attention()
    eval()