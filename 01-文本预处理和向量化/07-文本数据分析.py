# 导入必备工具包
from itertools import chain

import jieba
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(32, 6))

# 1 设置显示风格plt.style.use('fivethirtyeight')
plt.style.use('fivethirtyeight')

# 2 pd.read_csv 读训练集 验证集数据
train_data = pd.read_csv(filepath_or_buffer='data/train.tsv', sep='\t')
dev_data = pd.read_csv(filepath_or_buffer='data/dev.tsv', sep='\t')

# 获取句子长度分布
train_data["sentence_len"] = list(map(lambda x: len(jieba.lcut(x)), train_data['sentence']))

# 获取测试集的句子长度分布
dev_data["sentence_len"] = list(map(lambda x: len(jieba.lcut(x)), dev_data['sentence']))

# 思路分析 : 获取标签数量分布
# 0 什么标签数量分布：求标签0有多少个 标签1有多少个 标签2有多少个
# 1 设置显示风格plt.style.use('fivethirtyeight')
# 2 pd.read_csv(path, sep='\t') 读训练集 验证集数据
# 3 sns.countplot() 统计label标签的0、1分组数量
# 4 画图展示 plt.title() plt.show()
# 注意1：sns.countplot()相当于select * from tab1 group by
def dm_label_sns_countplot():
    # 3 sns.countplot() 统计label标签的0、1分组数量
    sns.countplot(x='label', data=train_data, hue='label')

    # 4 画图展示 plt.title() plt.show()
    plt.title('train_label')
    plt.show()

    # 验证集上标签的数量分布
    # 3-2 sns.countplot() 统计label标签的0、1分组数量
    sns.countplot(x='label', data=dev_data, hue='label')

    # 4-2 画图展示 plt.title() plt.show()
    plt.title('dev_label')
    plt.show()

    # 将 label 同时传给 x 和 hue，Seaborn 会根据类别自动应用默认调色盘
    # 如果你觉得默认颜色不好看，可以使用 palette 参数。常用的调色盘有 "viridis", "Set2", "pastel" 等
    sns.countplot(x='sentence_len', data=train_data, hue='sentence_len', palette='Set2')
    plt.title('train_sentence_len')
    plt.show()

    # 绘制数据长度分布图-曲线图
    sns.displot(x='sentence_len', data=train_data, kde=True)
    plt.title('train_sentence_len')
    plt.show()

    sns.countplot(x='sentence_len', data=dev_data, hue='sentence_len', palette='viridis')
    plt.title('dev_sentence_len')
    plt.show()
    # 绘制数据长度分布图-曲线图
    sns.displot(x='sentence_len', data=dev_data, kde=True)
    plt.title('dev_sentence_len')
    plt.show()


def dm_sns_stripplot():
    # 4 统计正负样本长度散点图 （对train_data数据，按照label进行分组，统计正样本散点图）
    sns.stripplot(x='label', y='sentence_len', data=train_data, jitter=True)
    plt.title('train_label_sentence_len')
    plt.show()

    # 5 统计正负样本长度散点图 （对dev_data数据，按照label进行分组，统计正样本散点图）
    sns.stripplot(x='label', y='sentence_len', data=dev_data, jitter=True)
    plt.title('dev_label_sentence_len')
    plt.show()

# 获取不同词汇总数统计
def dm_word_count():
    print(set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence']))))
    print(set(chain(*map(lambda x: jieba.lcut(x), dev_data['sentence']))))

    # train_data['word_count'] = list(map(lambda x: len(jieba.lcut(x)), train_data['sentence']))

if __name__ == '__main__':
    # dm_label_sns_countplot()
    # dm_sns_stripplot()
    dm_word_count()
