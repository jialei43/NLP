from itertools import chain

import jieba.posseg as pseg
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd


# 获取文本中的形容词
def get_adjective(text):
    words = pseg.lcut(text)
    adjectives = []
    for word, flag in words:
        if flag == 'a':  # a: 形容词
            adjectives.append(word)
    return adjectives


# 绘制词云
def draw_wordcloud(word_list):
    # 生成词云
    wordcloud = WordCloud(font_path='data/simhei.ttf', background_color='white').generate(' '.join(word_list))
    plt.imshow(wordcloud)
    # plt.axis('off')
    plt.show()


# 读取数据
def read_data():
    # 读取数据
    train_data = pd.read_csv(filepath_or_buffer='data/train.tsv', sep='\t')
    # 获取正样本
    sentence_lable_one = train_data[train_data['label'] == 1]['sentence']
    # 迭代获取正样本
    map_sentence = chain(*map(lambda x: get_adjective(x), sentence_lable_one))
    # 绘制词云
    draw_wordcloud(map_sentence)

    # 获取负样本
    sentence_lable_zero = train_data[train_data['label'] == 0]['sentence']
    # 迭代获取负样本
    # 这是一个非常典型的 Python 陷阱。代码无法进入 draw_wordcloud 的根本原因在于：map 函数在 Python 3 中是“惰性求值”（Lazy Evaluation）的
    # Python 3 中，map 不会立即执行函数，而是返回一个迭代器（Iterator）。
    # 如果你只是调用了 map(lambda x: draw_wordcloud(x), ...)，而没有去“消耗”或“遍历”这个迭代器，里面的代码就永远不会被执行。这就好比你写了一张购物清单（map对象），但你没拿着清单去超市买东西，所以抽屉里依然是空的ence_lable_zero))
    # 绘制词云
    map_sentence = chain(*map(lambda x: get_adjective(x), sentence_lable_zero))

    draw_wordcloud(map_sentence)






if __name__ == '__main__':
    # text = "房间里有电脑，虽然房间的条件略显简陋，但环境、服务还有饭菜都还是很不错的。如果下次去无锡，我还是会选择这里的。
    # draw_wordcloud(list(train_data["sentence"][0]))
    read_data()
