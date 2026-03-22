import fasttext
def dm_fasttext_train_save_load():
    """
    使用word2vec进行词向量生成
    :return:
    """
    # 1 使用train_unsupervised(无监督训练方法) 训练词向量
    mymodel = fasttext.train_unsupervised(input='data/fil9', model='cbow', dim=200, epoch=1,ws=5)
    print('训练词向量 ok')


    # 2 save_model()保存已经训练好词向量
    # 注意，该行代码执行耗时很长
    mymodel.save_model(path="./data/fil9.bin")
    print('保存词向量 ok')



def dm_fasttext_predict():
    """
    使用word2vec进行词向量测试
    :return:
    """
    # 3 模型加载
    model = fasttext.load_model(path='./data/fil9.bin')
    print('加载词向量 ok')

    # 打印zero的词向量
    # print(model.get_word_vector('zero'))
    # 打印最接近的5个单词
    print(model.get_nearest_neighbors('student', k=5))

if __name__ == '__main__':
    # dm_fasttext_train_save_load()
    dm_fasttext_predict()