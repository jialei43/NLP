import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


# embedding 词嵌入
class EmbeddingLayer(nn.Module):
    """
    Embedding层
    """
    def __init__(self,vocab_size,d_model,dropout_p=0):
        super().__init__()
        """
        初始化
        Args:
            vocab_size: 词典大小
            d_model: 词向量维度
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        """
        前向传播
        Args:
            x: 输入数据
        Returns:
            out: 输出数据
        """
        out = self.embedding(x)*math.sqrt(self.d_model)
        out = self.dropout(out)
        return out

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout_p,max_len=5000):
        super().__init__()
        """
        初始化
        Args:
            d_model: 词向量维度数
            dropout_p: 随机失活的概率，防止过拟合
            max_len: 句子最大
        """
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_p)

        # 定义一个位置编码矩阵
        pe = torch.ones(size=(max_len, d_model))
        # 创建一个词向量索引位置的序列
        # 将一维张量升维，与pe的维度匹配，以便进行广播
        pos = torch.arange(0, max_len).unsqueeze(dim=1)

        # 计算词索引的向量维度
        # 计算_2i,计算位置编码值时, 分母幂的2i值  10000**(2i/d_model)
        _2i = torch.arange(0, d_model, step=2).float()
        # 计算位置编码值，奇数为sin, 偶数为cos
        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # 将二维的位置编码矩阵pe在0轴升维,增加batch的维度，以便后续数据，当前形状是(max_len, d_model)
        pe = pe.unsqueeze(dim=0)
        # 将位置编码矩阵注册为模型的缓冲区（buffer），
        # 它不会被当作可训练参数，但会随模型一起保存和加载。
        self.register_buffer('pe', pe)
    def forward(self,x):
        """
        前向传播
        Args:
            x: 输入数据
        Returns:
            out: 输出数据
        """
        # x->词嵌入层后的词向量->(batch_size, seq_len, d_model)
        # self.pe -》位置编码-》(1, max_len, d_model)
        # self.pe[:, :x.shape[1], :]-》(1, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :]
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    vocab_size = 100
    d_model = 20
    dropout_p = 0
    embedding = EmbeddingLayer(vocab_size,d_model)
    out = embedding(torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))
    print(out)
    print(out.shape)
    print('-'*34)

    # 创建数据x[1,100,20], 给数据x添加位置特征  [1,100,20] ---> [1,100,20]
    input = torch.zeros(1, vocab_size)
    # # 计算位置编码
    # encoding = PositionalEncoding(d_model, 0)
    # out = encoding(input)
    # print(out.shape)
    # print(out)
    #
    # # 3 画图 绘制pe位置矩阵的第4-7列特征曲线
    # plt.figure(figsize=(20, 10))
    # # 第0个句子的，所有单词的，绘制4到8维度的特征 看看sin-cos曲线变化
    # plt.plot(np.arange(vocab_size), out[0, :, 4:8].detach().cpu().numpy())
    # plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    # plt.show()
    embedding1 = embedding(input.long())

    emb = PositionalEncoding(d_model, dropout_p)

    y = emb(embedding1)
    print('y--->', y.shape)

    # 3 画图 绘制pe位置矩阵的第4-7列特征曲线
    plt.figure(figsize=(20, 10))
    # 第0个句子的，所有单词的，绘制4到8维度的特征 看看sin-cos曲线变化
    plt.plot(np.arange(vocab_size), y[0, :, 4:8].detach().numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()
