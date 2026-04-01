import torch
from torch import nn
from data_processing import data_loader, device,PAD_token,MAX_LENGTH


class EncoderRNN(nn.Module):
    def __init__(self, eng_vocab_size, hidden_size):
        """
        初始化
        Args:
            eng_vocab_size: 英语词典大小
            hidden_size: 隐藏层维度
            你的理解已经非常接近本质了，但有一个关键点需要微调：词向量的维度（数据）并不是随机生成的，而是模型学习出来的。我们可以把 nn.Embedding 想象成一张不断进化的“电码表”。
            1. 结构与查找逻辑（你的理解是正确的）位置对应：你说得对，Embedding 层就是一个矩阵。输入单词的“索引”（Index）就是“行号”。
            查找过程：当我们输入索引 $i$ 时，Embedding 只是简单地把矩阵中第 $i$ 行的那个张量（向量）“拿出来”。这个过程不需要复杂的计算，仅仅是查表。
            2. 向量的值是怎么来的？（关键修正）你说“与词向量的值无关”，这在模型刚初始化时是对的，但在训练后就不对了。
            初始化阶段：当你刚创建 nn.Embedding 时，里面的数值确实是随机生成的（通常符合正态分布或均匀分布）。
            此时，两个意思相近的词（比如“猫”和“狗”）对应的向量确实毫无关系。
            训练阶段（学习过程）：随着训练的进行，这些向量的值会通过反向传播（Backpropagation）不断更新。如果模型发现每次出现“猫”和“狗”时，上下文都很相似，它就会自动调整这两行向量的数值，让它们在空间中靠得更近。
            最终结果：训练完成后，这些向量的值承载了语义信息。
        """
        super(EncoderRNN, self).__init__()
        self.eng_vocab_size = eng_vocab_size
        self.hidden_size = hidden_size
        # emb
        self.emb = nn.Embedding(num_embeddings=self.eng_vocab_size, embedding_dim=self.hidden_size, padding_idx=PAD_token)

        # gru
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

    def forward(self, x, h0):
        # print(x)
        # 嵌入:[1, seq_len]
        x = self.emb(x)
        # x :[1, seq_len, hidden_size]
        # h0 : [1,1,hidden_size]
        # gru处理
        # x = x.transpose_(0,1)
        out, hn = self.gru(x, h0)
        # out: [1, seq_len, hidden_size]
        # hn : [1,1,hidden_size]
        # 返回
        return out, hn

    def initHidden(self, batch_size):
        # 初始化隐藏状态
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        初始化
        Args:
            hidden_size: 隐藏层维度
            output_size: 输出维度  词向量大小
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 定义嵌入层 [词向量大小=输出维度，嵌入维度(词向量维度)=隐藏层维度(这俩个是超参数可以随便设置，目前设置为一样)]
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size, padding_idx=PAD_token)
        # 定义GRU [嵌入维度(词向量维度)，隐藏层维度]
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # 输出层 [隐藏层维度，输出维度]
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # 激活函数
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """

        :param input:
        :param hidden:
        :return:
        """
        # 目标语言嵌入
        input = self.embed(input)
        # 加入激活
        input = torch.relu(input)
        # out [1,seq_len,hidden_size] seq_len=1
        # gru处理
        out, hn = self.gru(input, hidden)
        # 输出层处理
        out = self.fc(out[0])
        # out [1,fre_word_num]
        # 加激活logsofmax
        out = self.softmax(out)
        return out, hn





class AttentionDecoderRNN(nn.Module):
    def __init__(self, fre_vocab_size, hidden_size, max_length, dropout_p=0.1, ):
        """
        初始化
        Args:
            fre_vocab_size: 特征维度（词向量维度，隐藏状态维度）
            hidden_size: 隐藏层维度
            max_length: 最长序列长度
            dropout_p: 随机失活概率
        """
        super(AttentionDecoderRNN, self).__init__()
        # 隐藏层维度
        self.hidden_size = hidden_size
        # 最长序列长度
        self.max_length = max_length
        # 随机失活概率
        self.dropout_p = dropout_p
        # 词向量维度
        self.fre_vocab_size = fre_vocab_size
        # 嵌入层 [词向量大小，嵌入维度(词向量维度)]
        self.embedding = nn.Embedding(self.fre_vocab_size, self.hidden_size, padding_idx=PAD_token)
        # 注意力层 [隐藏层维度*2，最长序列长度]
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # 注意力融合(线性层) [隐藏层维度*2，词向量维度]
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # 惩罚系数 随机失活
        self.dropout = nn.Dropout(self.dropout_p)
        # 模型层 GRU [(词向量维度)，隐藏层维度]
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 输出层 [隐藏层维度，词向量大小]
        self.out = nn.Linear(self.hidden_size, self.fre_vocab_size)
        # 激活函数
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, fre_input, hidden, encoder_outputs):
        """
        :param fre_input: Q->法语的文本 [batch_size,seq_len,hidden_size]
        :param hidden:  K->解码器的隐藏状态 [batch_size(1),seq_len(1),hidden_size]
        :param encoder_outputs: V->encoder的输出结果 [seq_len,hidden_size]
        :return:
        """
        # 嵌入层 Q: [batch_size, 1, hidden_size]
        Q = self.embedding(fre_input)
        # dropout Q: [batch_size, 1, hidden_size]
        Q = self.dropout(Q)
        # 拼接 Q+K [batch_size, seq_len, hidden_size*2]

        # hidden: hidden: [1, batch_size, hidden_size] -> [batch_size, 1, hidden_size]
        hidden_for_attn = hidden.permute(1, 0, 2)

        # X: [batch_size, 1, hidden_size * 2]
        X = torch.cat((hidden_for_attn, Q), -1)
        # attn_weights: [batch_size, 1, max_length]
        # 注意：这里必须是 dim=2，因为我们要对 max_length 个权重做归一化
        # self.attn 是一个线性层 nn.Linear(hidden_size * 2, self.max_length)。
        # 计算前：[batch_size, 1, hidden_size * 2]
        # 计算后：[batch_size, 1, max_length]
        attn_weights = torch.softmax(self.attn(X), dim=-1)
        # 4. C: [batch_size, 1, max_length] * [batch_size, max_length, hidden_size]
        # 结果 C: [batch_size, 1, hidden_size]
        C = torch.bmm(attn_weights, encoder_outputs)
        # 拼接 Q+C [batch_size, seq_len, hidden_size*2]
        output = torch.cat((Q, C), -1)
        # 注意力融合 (线性层) [batch_size, seq_len, hidden_size]
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # [batch_size, seq_len, hidden_size]经过[hidden_size * 2, hidden_size(词向量维度)]
        # 结果 output: [batch_size, seq_len, hidden_size(词向量维度)]
        output = torch.relu(self.attn_combine(output))
        # 6. GRU 和 输出层
        # 注意：如果 GRU 设置了 batch_first=True，hidden 依然维持 [1, batch, hidden]
        # [batch_size, seq_len, hidden_size(词向量维度)]经过GRU [hidden_size(词向量维度), hidden_size]
        # 输出结果 output: [batch_size, seq_len, hidden_size]
        output, hidden = self.gru(output, hidden)
        # output 从 [batch, 1, hidden] 降维为 [batch, hidden]
        output = output.squeeze(1)
        # 经过out = nn.Linear(self.hidden_size, self.fre_vocab_size)
        # 转换为 [batch,vocal_size] 供loss计算
        output = self.out(output)
        # 激活函数 [batch_size, seq_len, vocab_size]
        output = self.logsoftmax(output)
        return output, hidden, attn_weights


if __name__ == '__main__':
    # 加载数据
    dataloader, eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num = data_loader()
    # 模型实例化
    encoder_model = EncoderRNN(eng_vocab_size=eng_word_num, hidden_size=32)
    encoder_model.to(device)

    # 加载数据
    # dataloader, eng_word_num, eng_idx2word, eng_word2idx, fre_word2idx, fre_idx2word, fre_word_num = data_loader()
    # 模型实例化
    atten_decoder_model = AttentionDecoderRNN(fre_vocab_size=fre_word_num, hidden_size=32,max_length=MAX_LENGTH + 1, dropout_p=0.1)
    atten_decoder_model.to(device)

    # 遍历数据
    for x, y in dataloader:
        # out: [1, seq_len, hidden_size]
        # hn: [1, 1, hidden_size]
        # 初始化隐藏状态
        h0 = encoder_model.initHidden(x.size(0))
        out, hn = encoder_model(x, h0)
        print(x.shape)
        print(y.shape)
        print(out.shape)
        print(hn.shape)
        # 构建 V (encoder_output)
        # encoder_output = torch.zeros(x.size(0), MAX_LENGTH, 32, device=device)
        # for i in range(min(out.shape[1], MAX_LENGTH)):
        #     encoder_output[:, i, :] = out[:, i, :]
        # print(torch.equal(encoder_output, out))

        encoder_output = out
        decoder_hidden = hn

        # 解码器按时间步解码
        for i in range(y.shape[1]):
            # 取当前时间步的结果，保持 batch 维度
            # y[:, i]：y是当前Batch的目标语言张量，形状为[batch_size, seq_len]。
            # •: 表示选取所有的Batch（即所有8个样本）。
            # •    i表示选取当前时间步（第 $i$ 个单词）的索引。
            # •    结果形状：由[8, 12]降维成[8]，这是一个一维向量。
            # .unsqueeze(1)：
            # •    这个方法在索引为1的位置增加一个维度。
            # •    结果形状：由[8]变为[8, 1]。
            decoder_input = y[:, i].unsqueeze(1)  # [batch_size, 1]
            decoder_input = decoder_input.to(device)
            # 解码器解码
            out, decoder_hidden, atten_weight = atten_decoder_model(decoder_input, decoder_hidden, encoder_output)
            # 解码器输出
            print(out.shape)
            print(decoder_hidden.shape)
            print(f'attn_weights:{atten_weight}')
            print(atten_weight.shape)
            print('-' * 34)
