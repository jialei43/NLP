import math

import torch
from torch import nn

from input import EmbeddingLayer,PositionalEncoding


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
        # 多头的q,k，v的获取
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

