import torch
import torch.nn as nn

seq_len = 1
batch_size =5
input_size = 3
hidden_size = 8
# 单向lstm
def lstm( num_layers, bidirectional):
    """

    :param input_size: 词向量维度
    :param hidden_size: 隐藏状态维度
    :param num_layers: RNN堆叠的层数
    :param bidirectional: 是否双向
    :return:
    """
    # 初始化LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
    # 输入数据
    x = torch.randn(seq_len, batch_size, input_size)

    lstm_out, (h_n, c_n) = lstm(x)
    # bidirectional=true时，代表开启双向lstm，num_directions=2，
    # 双向会合并，所以只影响隐藏层数和隐藏状态的维度
    # 输出维度 seq_len, batch, hidden_size * num_directions
    print('lstm_out->', lstm_out.shape)
    # 隐藏层状态维度 num_layers * num_directions, batch, hidden_size
    print('h_n->',h_n.shape)
    # 细胞状态维度 num_layers * num_directions, batch, hidden_size
    print('c_n->',c_n.shape)


if __name__ == '__main__':
    lstm( num_layers=1, bidirectional=False)
    print('-'*34)
    lstm( num_layers=1, bidirectional=True)