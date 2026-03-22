import jieba
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from torch import nn

# 1 对句子分词 word_list
sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
sentence2 = "我爱自然语言处理"
sentences = [sentence1, sentence2]
word_list = []

# 分词
for sentence in sentences:
    word_list.append(jieba.lcut(sentence))


# 构建词表
# Tokenizer 实例化 会自动对词表进行去重，生成index_word word_index
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=word_list)
print(f'词汇表 word2id：{tokenizer.word_index}')
print(f'词汇表id2word：{tokenizer.index_word}')

# 获取词频
print(f'词频：{tokenizer.word_counts}')
# 获取词汇表大小
vocab_size = len(tokenizer.word_index.values())
print(f'词汇表大小：{vocab_size}')

# 词料库
word_sequences = tokenizer.texts_to_sequences(texts=word_list)

# 构建 词向量
embedding = nn.Embedding(vocab_size, 10)
for idx in range(len(tokenizer.index_word)):
    # embedding的输入是索引，所以从索引0开始
    tmpvec = embedding(torch.tensor(idx))
    # 打印
    print(f'{tokenizer.index_word[idx+1]} 的词向量是：{tmpvec.detach().numpy().round(4)}')

