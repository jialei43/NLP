# 数据清洗 data dataset dataloader
import re

import torch
from numpy import dtype
from torch import tensor
from torch.utils.data import Dataset, DataLoader

# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
# device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# 起始标志 SOS->Start Of Sequence
SOS_token = 0
# 结束标志 EOS->End Of Sequence
EOS_token = 1
# 用于设置每个句子样本的中间语义张量c长度都为10。
MAX_LENGTH = 7
#
PAD_token = 2
# 数据文件路径
data_path ='../data/eng-fra-v2.txt'

# 文本清洗工具函数
def normalizeString(s: str):
    """字符串规范化函数, 参数s代表传入的字符串"""
    # 将字符串转换为小写并去除首尾空白字符
    s = s.lower().strip()
    # 在标点符号(.!?)前添加空格，使其与单词分开。
    # \1 代表 捕获的标点符号，即 ., !, ? 之一。
    s = re.sub(r"([.!?])", r" \1", s)
    # 移除非字母字符和标点符号之外的所有字符，用空格替换
    s = re.sub(r"[^a-z.!?]+", r" ", s)
    # 返回规范化后的字符串
    return s
def read_data(path):
    """
    获取数据，构建词表
    :param path: 文件路径
    :return: 词表（英文，发文，词表大小），数据（英文，法文）
    """
    with open(path, 'r', encoding='utf-8') as f:
        # 创建两个列表，分别存储英文和法文数据
        data_sentences = []
        # 逐行读取数据
        num = 0
        for line in f.readlines():
            # 划分数据
            data_pair = line.strip().split('\t')
            # 规范化数据
            data_pair_temp = [normalizeString(sentence) for sentence in data_pair]
            # print(f'data_pair_temp:{data_pair_temp}')

            # 添加到列表中
            data_sentences.append(data_pair_temp)
            # print(data_sentences)
            # break

    # 构建词表
    # 英文
    eng_word2idx = {"SOS": SOS_token, "EOS": EOS_token, "PAD": PAD_token}
    # 法文
    fre_word2idx = fre_word2idx = {"SOS": SOS_token, "EOS": EOS_token, "PAD": PAD_token}
    # 词表起始加入索引大小
    eng_index = 3
    fre_index = 3
    # 遍历数据
    for pair in data_sentences:
        for index, sentence in enumerate(pair):
            for word in sentence.split(' '):
                if index == 0:
                    if word not in eng_word2idx:
                        eng_word2idx[word] = eng_index
                        eng_index += 1

                else:
                    if word not in fre_word2idx:
                        fre_word2idx[word] = fre_index
                        fre_index += 1


    # print(eng_word2idx)
    # 构建词表：id word
    eng_idx2word = {v: k for k, v in eng_word2idx.items()}
    # print(eng_idx2word)
    # print(fre_word2idx)
    fre_idx2word = {v: k for k, v in fre_word2idx.items()}
    # print(fre_idx2word)
    # 返回结果
    return eng_word2idx, eng_idx2word, eng_index, fre_word2idx, fre_idx2word, fre_index, data_sentences


class pairs_dataset(Dataset):
    """
    数据集
    """
    def __init__(self, data_pairs,eng_word2idx,fre_word2idx):
        super().__init__()
        """
        初始化
        Args:
            data_sentences: 数据
        """
        #  数据
        self.data_pairs = data_pairs
        # 词表 英文
        self.eng_word2idx = eng_word2idx
        # 词表 法文
        self.fre_word2idx = fre_word2idx
        # 样本数量
        self.sample_num = len(data_pairs)

    def __len__(self):
        """
        数据集大小
        Returns:
            数据集大小
        """
        return self.sample_num
    def __getitem__(self, index):
        """
        获取数据
        Args:
            index: 索引

        Returns:
            数据
        """
        index = min(max(index, 0), self.sample_num - 1)
        x_tensor = self.data_pairs[index][0]
        y_tensor = self.data_pairs[index][1]

        # 转换为数值向量
        # 分割出x
        x_tensor = [self.eng_word2idx[word] for word in x_tensor.split(' ')]
        # 文本规范，x_tensor 设置做大长度，不足补充，太长截断
        if len(x_tensor) < MAX_LENGTH:
            x_tensor += [PAD_token] * (MAX_LENGTH - len(x_tensor))
        else:
            x_tensor = x_tensor[:MAX_LENGTH]
        x_tensor = torch.tensor(x_tensor + [EOS_token], dtype=torch.long, device=device)

        # 分割出y
        y_tensor = [self.fre_word2idx[word] for word in y_tensor.split(' ')]
        # 文本规范，y_tensor 设置做大长度，不足补充，太长截断
        if len(y_tensor) < MAX_LENGTH:
            y_tensor += [PAD_token] * (MAX_LENGTH - len(y_tensor))
        else:
            y_tensor = y_tensor[:MAX_LENGTH]
        y_tensor = torch.tensor([SOS_token]+y_tensor + [EOS_token], dtype=torch.long, device=device)

        return x_tensor, y_tensor


def data_loader():
    eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num, data_pairs = read_data(
        data_path)
    # print(eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num, data_pairs)
    pairs = pairs_dataset(data_pairs, eng_word2idx, fre_word2idx)
    # print(pairs.__getitem__(0))
    dataloader = DataLoader(dataset=pairs, batch_size=64, shuffle=True)
    return dataloader,eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num






if __name__ == '__main__':
    # print(normalizeString('Adsxs ds & sded? '))
    # print(read_data(data_path))
    eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num, data_pairs = read_data(
        data_path)
    print(eng_word2idx)
    print(fre_word2idx)
    pairs = pairs_dataset(data_pairs, eng_word2idx, fre_word2idx)
    print(pairs.__getitem__(0))

    dataloader = data_loader()
    for x, y in dataloader:
        print(x)
        print(y)
        break