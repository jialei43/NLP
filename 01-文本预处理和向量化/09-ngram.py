from sklearn.feature_extraction.text import CountVectorizer

input_list = [1, 3, 2, 1, 5, 3]
corpus = [
        "I love NLP.",
        "NLP is fun!",
        "I study natural language processing one two language processing"]
ngram_range =2
def get_ngram(input_list, ngram_range):
    """
    获取Ngram
    :param input_list:
    :param ngram_range:
    :return:
    """
    # 合并列表
    list = set(zip(*[input_list[i:] for i in range(ngram_range)]))
    print(list)

def skelearn_ngram():
    # 实例化
    # ngram_range: 获取的Ngram范围ngram_range=(1, 3) 会生成1个单词的词汇、2个单词的词汇、3个单词的词汇
    vectorizer = CountVectorizer(ngram_range=(ngram_range, ngram_range))
    # 训练及测试
    x = vectorizer.fit_transform(corpus)
    # 输出结果
    print(vectorizer.get_feature_names_out())
    print(x.toarray())
    print(x)
if __name__ == '__main__':
    # get_ngram(input_list, ngram_range)
    skelearn_ngram()