import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class EmbeddingLayer(nn.Module):
    """
    Embedding层
    """
    def __init__(self, vocab_size, m_dim):
        """
        初始化
        Args:
            vocab_size: 词典大小
            m_dim: 词向量维度
        """
        super(EmbeddingLayer, self).__init__()
        self.m_dim = m_dim
        self.vocab_size = vocab_size
        # 词向量层
        self.embedding = nn.Embedding(vocab_size, m_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 将x传给self.embed并与根号下self.d_model相乘作为结果返回
        # 词嵌入层的权重通常神经网络参数初始化默认使用xiaver, 值一般在(-0.1, 0.1)之间, 导致嵌入后的向量幅度较小。
        # 词向量结果和位置编码信息相加, 位置编码信息值在(-1,1)之间
        # x经过词嵌入后乘以sqrt(d_model)来增大x的值, 与位置编码信息值量纲[-1,1]差不多, 确保两者相加时信息平衡。
        x = self.embedding(x)*math.sqrt(self.m_dim)
        x = self.dropout(x)
        return x

# Transformer模型中的位置编码（Positional Encoding），用于为输入序列中的每个位置添加位置信息，
# Transformer的自注意力机制本身无法感知序列的顺序关系
class PositionalEncoding(nn.Module):
    # todo:1- init构造函数
    def __init__(self, d_model, dropout_p, max_len=5000):
        """
        :param d_model: 词向量维度数
        :param dropout_p: 随机失活的概率，防止过拟合
        :param max_len: 句子最大长度
        """
        super().__init__()
        # 随机失活层，训练阶段随机丢弃部分神经元
        self.dropout = nn.Dropout(p=dropout_p)
        # 创建一个形状为(max_len, d_model)的全零矩阵，用于存储位置编码 pe。
        pe = torch.zeros(size=(max_len, d_model))
        # print('pe--->', pe.shape)
        # 获取句子中词的位置索引 [0,1,2,3,4, ...max_len-1] pos ->形状是（max_len,）
        # 将pos在1轴升维 [[0],[1],[2],..], 该数据中第i行表示第i个词的位置i  pos 形状是（max_len,1）
        # 增加一维与pe的维度匹配，以便进行广播
        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        # 创建偶数索引
        # 计算_2i, 计算位置编码值时, 分母幂的2i值  10000**(2i/d_model)
        _2i = torch.arange(0, d_model, step=2).float()
        # 计算位置编码值, 奇数位sin, 偶数位cos
        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # 将二维的位置编码矩阵pe在0轴升维,增加batch的维度，以便后续数据
        pe = pe.unsqueeze(dim=0)
        # print('pe--->', pe.shape, pe)
        # 将位置编码矩阵注册为模型的缓冲区（buffer），
        # 它不会被当作可训练参数，但会随模型一起保存和加载。
        self.register_buffer('pe', pe)

    # todo:2- forward方法, 计算
    def forward(self, x):
        # x->词嵌入层后的词向量->(batch_size, seq_len, d_model)
        # self.pe -》位置编码-》(1, max_len, d_model)
        # self.pe[:, :x.shape[1], :]-》(1, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


# 前馈全连接层PositionwiseFeedForward实现思路分析
# 1 init函数  (self,  d_model, d_ff, dropout=0.1):
    # 定义线性层self.linear1 self.linear2, self.dropout层
# 2 forward(self, x)
    # 数据经过self.w1(x) -> F.relu() ->self.dropout() ->self.w2 返回



if __name__ == '__main__':
    d_model = 20
    dropout_p = 0
    vocab_size = 100
    embedding = EmbeddingLayer(vocab_size, d_model)
    out = embedding(torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))
    print(out)
    print(out.shape)
    print('-'*34)

    pe = PositionalEncoding(d_model,dropout_p)
    pe_out = pe(out)
    print(pe_out)
    print(pe_out.shape)

    # 2 创建数据x[1,100,20], 给数据x添加位置特征  [1,100,20] ---> [1,100,20]
    input = torch.zeros(1, vocab_size, d_model)
    emb = PositionalEncoding(d_model, dropout_p)
    out = emb(input)
    y = emb(out)
    print('y--->', y.shape)

    # 3 画图 绘制pe位置矩阵的第4-7列特征曲线
    plt.figure(figsize=(20, 10))
    # 第0个句子的，所有单词的，绘制4到8维度的特征 看看sin-cos曲线变化
    plt.plot(np.arange(vocab_size), y[0, :, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()
