import copy
import math

import torch
from torch import nn

from input import EmbeddingLayer, PositionalEncoding


def attention(query,key,value,mask=None,dropout=None):
    """
    计算注意力
    Args:
        query: 查询向量
        key: 关键词向量
        value: 值向量
        mask: 遮罩矩阵
        dropout: 随机失活层
    Returns:
        out: 输出向量
        attention_weight: 注意力权重
    """
    # 输入维度
    d_k = query.size(-1)
    # 矩阵乘法
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 遮罩矩阵
    if mask is not None:
        # 遮罩矩阵乘以-1e9
        scores = scores.masked_fill(mask == 0, -1e9)

    weight = torch.softmax(scores, dim=-1)

    if dropout is not None:
        weight = dropout(weight)

    # 注意力结果
    out = torch.matmul(weight, value)
    # 返回输出向量和注意力权重
    return out, weight

def dm_test_attention():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    # 输入部分的Embeddings类
    my_embeddings = EmbeddingLayer(vocab, d_model)
    x = my_embeddings(x)

    dropout_p = 0.1  # 置0概率为0.1
    max_len = 60  # 句子最大长度

    # 输入部分的PositionalEncoding类  编码位置矩阵
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(x)

    query = key = value = pe_result  # torch.Size([2, 4, 512])
    print('编码阶段 对注意力权重分布 不做掩码')

    attn1, p_attn1 = attention(query, key, value)
    print('注意力权重 p_attn1--->', p_attn1.shape, '\n', p_attn1)  # torch.Size([2, 4, 4])
    print('注意力表示结果 attn1--->', attn1.shape, '\n', attn1)  # torch.Size([2, 4, 512])


# 创建N个相同模块的深度拷贝，拷贝的层模块结构完全相同，但它们的参数是各自独立、完全不同的
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 多头自注意力
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        # 整除
        assert d_model % h == 0
        # 每个头的维度
        self.d_k = d_model // h
        # 头数
        self.h = h
        # 线性层
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        # 随机失活
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 掩码
        if mask is not None:
            # 扩张batch维度 【heads,seq_len_q,seq_len_k】
            mask = mask.unsqueeze(0)
        # 获取batch
        batch_size = query.size(0)
        # 多头的q,k，v的获取 [2,8,4,64]
        q, k, v = [model(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for model, x in
                   zip(self.linears, [query, key, value])]
        # 注意力
        atten, weight = attention(q, k, v, mask=mask)
        # 拼接
        out = atten.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # 失活
        out = self.dropout(out)
        # 输出层处理
        return self.linears[-1](out)

# 前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        # d_model  第1个线性层输入维度
        # d_ff     第2个线性层输出维度
        super(PositionwiseFeedForward, self).__init__()
        # 定义线性层w1 w2 dropout
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # 数据依次经过第1个线性层 relu激活层 dropout层，然后是第2个线性层
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x= (x-mean)/(std+self.eps)
        return self.gamma * x + self.beta

# 子层连接结构 子层(前馈全连接层 或者 注意力机制层)+ norm层 + 残差连接
# SublayerConnection实现思路分析
# 1 init函数  (self, size, dropout=0.1):
# 定义self.norm层 self.dropout层, 其中LayerNorm(size)
# 2 forward(self, x, sublayer) 返回+以后的结果
# 数据self.norm() -> sublayer()->self.dropout() + x
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_p=0.1):
        super(SublayerConnection, self).__init__()
        # 参数size 词嵌入维度尺寸大小
        # 参数dropout 置零比率
        self.size = size
        self.dropout_p = dropout_p
        # 定义norm层
        self.norm = LayerNorm(self.size)
        # 定义dropout
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, sublayer):
        # 参数x 代表数据
        # sublayer 函数入口地址 子层函数(前馈全连接层 或者 注意力机制层函数的入口地址)
        # 数据self.norm() -> sublayer() -> self.dropout() + x
        # Transformer的标准实现，通常效果最好
        # myres = x + self.dropout(sublayer(self.norm(x)))
        # 残差连接
        myres = self.norm(x + self.dropout((sublayer(x))))
        return myres

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_p=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout_p), 2)
        self.size = size
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

if __name__ == '__main__':
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    # 输入部分的Embeddings类
    my_embeddings = EmbeddingLayer(vocab, d_model)
    x = my_embeddings(x)

    dropout_p = 0.1  # 置0概率为0.1
    max_len = 60  # 句子最大长度

    # 输入部分的PositionalEncoding类
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    pe_result = my_pe(x)

    heads = 8
    Q = K = V = pe_result
    mask = torch.tril(torch.ones((8,4,4)))
    # atten: [batch,seq_len_q,d_k]
    # weight:[batch,seq_len_q,seq_len_k]
    # atten, weight = attention(Q, K, V)
    # print(atten.shape)
    # print(atten)
    # print(weight.shape)
    # print(weight)
    #
    # mask = torch.tril(torch.ones((2, 4, 4)))
    # print(mask)
    # atten, weight = attention(Q, K, V, mask)
    # print(atten.shape)
    # print(atten)
    # print(weight.shape)
    # print(weight)
    mha = MultiHeadedAttention(h=heads, d_model=d_model, dropout=dropout_p)
    out = mha(Q, K, V)
    print(out.shape)
    print(out)
    out = mha(Q, K, V,mask)
    print(out.shape)
    print(out)

    print('-'*34)
    # 测试前馈全链接层
    my_PFF = PositionwiseFeedForward(d_model=512, d_ff=2048, dropout_p=0.1)
    ff_result = my_PFF(out)
    print('x--->', ff_result.shape, ff_result)

    # 测试层归一化层
    print('-'*34)
    features = d_model = 512
    eps = 1e-6
    x = ff_result
    ln = LayerNorm(features, eps)
    ln_result = ln(x)
    print('规范化层:', ln_result.shape, ln_result)

    # 测试子层连接结构
    print('-'*34)
    size = 512
    head = 8
    x = pe_result
    mask = torch.tril(torch.ones(size=(8, 4, 4))).type(torch.uint8)
    # 多头自注意力子层
    self_attn = MultiHeadedAttention(head, d_model,dropout_p)
    sublayer = lambda x: self_attn(x, x, x, mask)
    # 子层连接结构
    sc = SublayerConnection(size, dropout_p)
    sc_result = sc(x, sublayer)
    print('sc_result.shape--->', sc_result.shape)
    print('sc_result--->', sc_result)

    print('-'*34)
    c = copy.deepcopy
    layer = EncoderLayer(size, c(self_attn), c(my_PFF), dropout_p)
    layer_result = layer(pe_result, mask)
    print('layer_result.shape--->', layer_result.shape)

    print('-'*34)
    encoder = Encoder(layer, N=6)
    encoder_result = encoder(pe_result, mask)
    print('encoder_result.shape--->', encoder_result.shape)


