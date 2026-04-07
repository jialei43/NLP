import math

import torch


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


