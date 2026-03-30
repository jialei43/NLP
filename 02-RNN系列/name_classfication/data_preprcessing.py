import string

import torch
from keras.preprocessing import sequence
from torch.utils.data import Dataset, DataLoader

# 词表
letters = string.ascii_letters + " ,.;'"
# 词表大小
letters_num = len(letters)
# print(letters_num)
# print(letters[5])

# 目标值
categories = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
              'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']

# 类别个数
categories_num = len(categories)


# print(categories_num)

def get_data(path=r'data/name_classfication.txt'):
    '''
    读取数据txt
    Args:
        path: 数据文件路径

    Returns:
        x_data: 姓名
        y_data： 国家
    '''
    # 初始化
    x_data = []
    y_data = []
    # 文件流
    with open(path, 'r', encoding='utf-8') as f:
        # # 读所有行，并遍历
        # for line in f.readlines():
        #     # 划分数据
        #     name, nation = line.strip().split('\t')
        #     # 添加到列表中
        #     x_data.append(name)
        #     y_data.append(nation)
        # 逐行读取
        while True:
            readline = f.readline()
            if not readline:
                break
            # 划分数据
            name, nation = readline.strip().split('\t')
            # 添加到列表中
            x_data.append(name)
            y_data.append(nation)
    # 返回
    return x_data, y_data

# 创建数据集
class NameClassDataSet(Dataset):
    """
    数据集
    """
    def __init__(self, x_data, y_data):
        super().__init__()
        """
        初始化
        Args:
            x_data: 姓名
            y_data: 国家
        """
        self.x_data = x_data
        self.y_data = y_data


    def __len__(self):
        """
        数据集大小
        Returns:
            数据集大小
        """
        return len(self.x_data)

    def __getitem__(self, index):
        """
        获取数据
        Args:
            index: 索引

        Returns:
            姓名
            国家
        """
        y = self.y_data[index]
        x = self.x_data[index]
        MAX_LENGTH = 10
        
        # 获取国家
        tensor_y = torch.tensor(categories.index(y),dtype=torch.long)
        # # 创建 姓名 对应的全0矩阵
        # tensor_x = torch.zeros(len(x), letters_num,dtype=torch.float32)
        # # 遍历姓名
        # for i, letter in enumerate(x):
        #     # 创建姓名对应矩阵 ONE-HOT
        #     tensor_x[i, letters.index(letter)] = 1
        # --- 修改处：统一长度为 MAX_LENGTH ---
        # 创建 [MAX_LENGTH, letters_num] 的全0矩阵
        tensor_x = torch.zeros(MAX_LENGTH, letters_num, dtype=torch.float32)

        # 遍历姓名，如果姓名超过 MAX_LENGTH 则截断，不足则保留为0（即Padding）
        for i, letter in enumerate(x[:MAX_LENGTH]):
            if letter in letters:
                tensor_x[i, letters.index(letter)] = 1

        return tensor_x, tensor_y
# 数据加载
def data_loader():
    # 获取数据
    x_data, y_data = get_data()
    # 创建数据集
    dataSet = NameClassDataSet(x_data, y_data)
    # 创建数据加载器
    # 批次大小 batch_size shuffle 是否打乱
    dataloader = DataLoader(dataSet, batch_size=64, shuffle=True)

    # 返回
    return dataloader


if __name__ == '__main__':
    x_data,y_data=get_data()
    # print(x_data)
    # print(y_data)

    # dataSet = NameClassDataSet(x_data, y_data)
    # print(dataSet[0][0].shape)
    # print(dataSet.__getitem__(0)[0].shape)
    loader = data_loader()
    for x, y in loader:
        print(x.shape)
        str_x = ''
        # 获取最后一维最大值所在索引
        indices = torch.argmax(x, dim=-1)
        # 将二维张量转换为一维的张量，然后遍历
        for value in indices.flatten():
            # print(value)
            # 获取字母，且拼接成字符串
            str_x+=letters[value]
        # 输出索引
        print(indices)
        # 输出姓名
        print(str_x)
        # print(x)
        # [1,len(x),57]
        # [bs,seq_len,input_size]
        # [bs,]
        print(y)
        # 输出国家
        print(categories[y])
        break
