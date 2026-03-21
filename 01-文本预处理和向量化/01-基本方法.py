import jieba

content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"

# 精确模式 精确模式分词： 试图将句子最精确地切分开，适合文本分析。
words = jieba.cut(content,cut_all=False)
print(words)
print(list(words))

words = jieba.lcut(content,cut_all=False)
print(f'精确模式:{words}')

# 全模式 将句子中所有可以成词的词语都扫描出来，速度非常快，但是不能消除歧义
words = jieba.lcut(content,cut_all=True)
print(f'全模式: {words}')

# 搜索引擎模式 在精确模式的基础上，对长词再次切分，进行细粒度分词，适合用于搜索引擎分词
words = jieba.lcut_for_search(content)
print(f'搜索引擎模式: {words}')

# 自定义词典 格式：词语 词频 词性
jieba.load_userdict("./userdict")
words = jieba.lcut(content)
print(f'自定义词典: {words}')