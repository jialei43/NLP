import numpy as np
import torch
from matplotlib import pyplot as plt

from model import EncoderRNN, AttentionDecoderRNN, MAX_LENGTH
from data_processing import device, normalizeString, data_loader, EOS_token, SOS_token


def seq2seq_eval(x,encoderRnn,atten_decoderRnn,fre_idx2word):
    # 初始化隐藏状态
    h0 = encoderRnn.initHidden(1)
    # encoder模型训练
    encoder_output, encode_hidden = encoderRnn(x,h0)
    # atten_decoderRnn模型训练
    encoder_outputs_c = encoder_output
    decoder_hidden = encode_hidden
    # 构建输入数据
    decoder_input = torch.tensor([[SOS_token]],device=device,dtype=torch.long)
    # 存储生成的翻译文本
    seq_list = []
    # atten 矩阵初始化
    # 词汇经过注意力层输出的权重矩阵[seq_len,max]
    atten_matrix = torch.zeros(MAX_LENGTH+2,MAX_LENGTH+2,device= device)
    for idx in range(MAX_LENGTH+2):
        output, hidden, attn_weights = atten_decoderRnn(decoder_input, decoder_hidden, encoder_outputs_c)
        # 填充weight
        atten_matrix[idx] = attn_weights
        # 获取最大的结果
        input_y = torch.argmax(output, dim=-1).view(1, -1)
        # 判断是否截止符
        if input_y.item() == EOS_token:
            seq_list.append('<EOS>')
            break
        else:
            # 填充到列表中
            seq_list.append(fre_idx2word[input_y.item()])

    return seq_list, atten_matrix[:idx + 1]


# 注意力可视化热力图
def plot_attention():
    # 加载数据
    dataloader, eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num = data_loader()

    # 模型实例化
    encoderRnn = EncoderRNN(eng_vocab_size=eng_word_num, hidden_size=32).to(device)
    atten_decoderRnn = AttentionDecoderRNN(fre_vocab_size=fre_word_num, hidden_size=32).to(device)

    # 模型加载
    encoderRnn.load_state_dict(torch.load(r'./model/encoderRnn.pth',map_location=device))
    atten_decoderRnn.load_state_dict(torch.load(r'./model/atten_decoderRnn.pth',map_location=device))

    # 设置为eval模式
    encoderRnn.eval()
    atten_decoderRnn.eval()

    # 数据
    x = 'she is beautiful like her mother .'
    x = normalizeString(x)

    # 数据处理
    x = [eng_word2idx[(word)] for word in x.split(' ')]

    # 文本长度规范 不足填充2 超出截断
    if len(x) < MAX_LENGTH:
        x += [2] * (MAX_LENGTH - len(x))
    else:
        x = x[:MAX_LENGTH]

    # 添加结束标志
    x.append(EOS_token)
    # 转换为张量
    x = torch.tensor(x,dtype=torch.long).to(device)
    # 添加起始标志
    x = torch.cat((torch.tensor([SOS_token]).to(device), x), dim=0).unsqueeze(0)


    # 模型推理
    fre_word_list, atten = seq2seq_eval(x,encoderRnn,atten_decoderRnn,fre_idx2word)
    # 创建热图，使用viridis颜色映射显示注意力权重
    fig, ax = plt.subplots()
    # cmap:指定一个颜色映射，将数据值映射到颜色
    # viridis:从深紫色（低值）过渡到黄色（高值），具有良好的对比度和可读性
    cax = ax.matshow(atten.cpu().detach().numpy(), cmap='viridis')
    # 添加颜色条
    fig.colorbar(cax)
    # 添加标签
    for (i, j), value in np.ndenumerate(atten.cpu().detach().numpy()):
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white')
    # 保存图像
    plt.savefig("s2s_attn.png")
    plt.show()


if __name__ == '__main__':
    plot_attention()