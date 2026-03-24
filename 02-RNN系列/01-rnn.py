import torch
from torch import nn
def rnn_num_layer_one():
    input_size = 10
    hidden_size = 8
    num_layers = 1
    seq_len = 1
    batch_size =9

    # 定义RNN模型
    # 输入数据维度: (seq_len, batch, input_size)
    rnn = nn.RNN(input_size, hidden_size, num_layers)

    # 输入数据
    # 输入数据维度: (seq_len, batch, input_size)
    input = torch.randn(seq_len, batch_size, input_size)

    # 初始化隐藏层状态
    # 隐藏层状态维度: (num_layers , batch, hidden_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    rnn_out, hidden = rnn(input, hidden)

    # 输出维度 (seq_len, batch, hidden_size)
    print(rnn_out.shape)
    print(rnn_out)
    print('-'*34)
    # 隐藏层状态维度 (num_layers, batch, hidden_size)
    print(hidden.shape)
    print(hidden)


def rnn_num_layer_two():
    input_size = 10
    hidden_size = 8
    num_layers = 2
    seq_len = 5
    batch_size = 9

    # 定义RNN模型
    # 输入数据维度: (seq_len, batch, input_size)
    rnn = nn.RNN(input_size, hidden_size, num_layers)

    # 输入数据
    # 输入数据维度: (seq_len, batch, input_size)
    input = torch.randn(seq_len, batch_size, input_size)

    # 初始化隐藏层状态
    # 隐藏层状态维度: (num_layers , batch, hidden_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    rnn_out, hidden = rnn(input, hidden)

    # 输出维度 (seq_len, batch, hidden_size)
    print(rnn_out.shape)
    print(rnn_out)
    print('-' * 34)
    # 隐藏层状态维度 (num_layers, batch, hidden_size)
    print(hidden.shape)
    print(hidden)

if __name__ == '__main__':
    # rnn_num_layer_one()
    rnn_num_layer_two()