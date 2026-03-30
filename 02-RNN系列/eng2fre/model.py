import torch
from torch import nn
from data_processing import data_loader,device

class EncoderRNN(nn.Module):
    def __init__(self, eng_vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.eng_vocab_size = eng_vocab_size
        self.hidden_size = hidden_size
        # emb
        self.emb = nn.Embedding(num_embeddings=self.eng_vocab_size, embedding_dim=self.hidden_size)

        # gru
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

    def forward(self, x, h0):
        print(x)
        # 嵌入:[1, seq_len]
        x = self.emb(x)
        # x :[1, seq_len, hidden_size]
        # h0 : [1,1,hidden_size]
        # gru处理
        out, hn = self.gru(x, h0)
        # out: [1, seq_len, hidden_size]
        # hn : [1,1,hidden_size]
        # 返回
        return out, hn

    def initHidden(self):
        # 初始化隐藏状态
        return torch.zeros(1, 1, self.hidden_size)

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
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
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



MAX_LENGTH=10
class AttentionDecoderRNN(nn.Module):
    def __init__(self, fre_vocab_size, hidden_size, max_length=MAX_LENGTH, dropout_p=0.1, ):
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
        self.embedding = nn.Embedding(self.fre_vocab_size, self.hidden_size)
        # 注意力层 [隐藏层维度*2，最长序列长度]
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # 注意力融合(线性层) [隐藏层维度*2，词向量大小]
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # 惩罚系数 随机失活
        self.dropout = nn.Dropout(self.dropout_p)
        # 模型层 GRU [(词向量维度)，隐藏层维度]
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
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
        # 嵌入层 [batch_size,seq_len,hidden_size]
        Q = self.embedding(fre_input)
        # dropout [batch_size,seq_len,hidden_size]
        Q = self.dropout(Q)
        # 拼接Q+K [batch_size,seq_len,hidden_size*2]
        X = torch.cat((hidden[0], Q[0]), -1)
        # 注意力层 相似度计算 [seq_len,max_length]
        attn_weights = torch.softmax(self.attn(X), dim=1)
        # 加权求和，求得注意力结构 [batch_size,seq_len,hidden_size]=[1,1,hidden_size]
        C = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 拼接Q+C [batch_size,seq_len,hidden_size]->[seq_len,hidden_size*2]
        output = torch.cat((Q[0], C[0]), -1)
        # 注意力融合(线性层) [batch_size,seq_len,hidden_size]
        output = torch.relu(self.attn_combine(output)).unsqueeze(0)
        # 模型层 GRU [batch_size,seq_len,hidden_size]
        output, hidden = self.gru(output, hidden)
        # 输出层 [batch_size,seq_len,hidden_size]
        output = self.out(output[0])
        # 激活函数 [batch_size,seq_len,hidden_size]
        output = self.logsoftmax(output)
        return output, hidden, attn_weights


if __name__ == '__main__':
    # 加载数据
    dataloader, eng_word_num, eng_idx2word, eng_word2idx, fre_word2idx, fre_idx2word, fre_word_num = data_loader()
    #模型实例化
    encoder_model = EncoderRNN(eng_vocab_size=eng_word_num, hidden_size=32)
    encoder_model.to(device)
    # 初始化隐藏状态
    h0 = encoder_model.initHidden()
    h0 = h0.to(device)
    # # 遍历数据
    for x, y in dataloader:
        # out: [1, seq_len, hidden_size]
        # hn: [1, 1, hidden_size]
        x = x.to(device)
        out, hn = encoder_model(x, h0)
        print(out.shape, hn.shape)
        break

    # 加载数据
    # dataloader, eng_word_num, eng_idx2word, eng_word2idx, fre_word2idx, fre_idx2word, fre_word_num = data_loader()
    # 模型实例化
    atten_decoder_model = AttentionDecoderRNN(fre_vocab_size=fre_word_num, hidden_size=32,dropout_p=0.1)
    atten_decoder_model.to(device)
    # 遍历数据
    for x, y in dataloader:
        # out: [1, seq_len, hidden_size]
        # hn: [1, 1, hidden_size]
        x = x.to(device)
        y = y.to(device)
        out, hn = encoder_model(x, h0)
        print(y)
        print(y.shape)
        # 全零
        encoder_output = torch.zeros(x.size(0),MAX_LENGTH,32)
        encoder_output = encoder_output.to(device)
        # 遍历数据
        for x, y in dataloader:
            # out: [1, seq_len, hidden_size]
            # hn: [1, 1, hidden_size]
            # 编码器进行编码
            x.to(device)
            y = y.to(device)
            out, hn = encoder_model(x, h0)
            # print(out.shape, hn.shape)
            # print(out)
            # print(hn)
            print(y)
            print(y.shape)
            # 全零
            encoder_output = torch.zeros(MAX_LENGTH, 32)
            # 遍历out
            for i in range(out.shape[1]):
                if i >= MAX_LENGTH:
                    break
                # 构建V
                encoder_output[i] = out[0][i]
            encoder_output.to(device)
            # 解码器按时间步解码
            for i in range(y.shape[1]):
                # 取当前时间步的结果，并添加batch维
                input = y[0][i].view(1, -1)
                # 解码器解码
                # out, hn = decoder_model(input, hn)
                input = input.to(device)
                out, hn, atten_weight = atten_decoder_model(input, hn, encoder_output)
                # 解码器输出
                print(out.shape)
                print(hn.shape)
                print(atten_weight)
                print(atten_weight.shape)

