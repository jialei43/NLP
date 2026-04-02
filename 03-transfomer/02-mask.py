import numpy as np
import torch
from matplotlib import pyplot as plt


# input = np.ones((3, 3))
# # print(input)
# triu = np.triu(input, k=1)
# print(triu)
# # print(triu.shape)
# print('-'*34)
#
# tril = np.tril(input, k=-1)
# print(tril)
def mask(size,k=0):
    """
    生成一个下三角矩阵
    :param size: 矩阵大小
    :return:
    """
    mask = np.triu(np.ones((1,size, size)).dtype(torch.uint8),k)
    # torch.tril(mask, k)
    return mask


out= mask(5, -1)
print(out)
plt.imshow(out[0])
plt.show()