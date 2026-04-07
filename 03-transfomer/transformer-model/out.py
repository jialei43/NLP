# 解码器类 Generator 实现思路分析
# init函数 (self, d_model, vocab_size)
# 定义线性层self.out
# forward函数 (self, x)
# 数据 torch.log_softmax(self.project(x), dim=-1)
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        # 参数d_model 线性层输入特征尺寸大小
        # 参数vocab_size 线性层输出尺寸大小
        super(Generator, self).__init__()
        # 定义线性层
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 数据经过线性层 最后一个维度归一化 log方式
        x = torch.log_softmax(self.out(x), dim=-1)
        return x

if __name__ == '__main__':
    # 实例化output层对象
    # 解码器预测词的输出维度
    d_model = 512
    # 词汇表中词数量
    vocab_size = 1000
    my_generator = Generator(d_model, vocab_size)

    # 准备模型数据
    x = torch.randn(2, 4, 512)

    # 数据经过out层
    gen_result = my_generator(x)
    print('gen_result--->', gen_result.shape, '\n', gen_result)