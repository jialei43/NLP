import torch.nn as nn
import torch

# 实例化
# input_size: 词向量的维度
# hidden_size : 隐藏状态的维度
# num_layer: RNN堆叠的层数
# bidirectional： 是否双向
gru = nn.GRU(input_size=3, hidden_size=4, num_layers=2, bidirectional=True)

# 输入
# x: [seq_len，bs,input_size]
# seq_len: 分词后句子的长度
# bs: 句子数量
# input_size：词向量的维度
x = torch.randn(1, 5, 3)
# h0:[num_layer,bs,hidden_size] 初始化为全0
# num_layer: GRU堆叠的层数
# bs: 句子数量
# hidden_size : 隐藏状态的维度
h0 = torch.zeros(4, 5, 4)
# 网络中
out, hn = gru(x, h0)
# out: [seq_len，bs,hidden_size*2]
# seq_len: 分词后句子的长度
# bs: 句子数量
# hidden_size : 隐藏状态的维度
# 1,5,4*2
print(out.shape)
#2*2,5,4
# hn:[num_layer*2,bs,hidden_size]
# num_layer: GRU堆叠的层数
# bs: 句子数量
# hidden_size : 隐藏状态的维度
print(hn.shape)
