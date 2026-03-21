import fasttext
def dm_fasttext_train_save_load():
    # 1 使用train_unsupervised(无监督训练方法) 训练词向量
    mymodel = fasttext.train_unsupervised(input='data/fil9', model='skipgram', dim=300, epoch=1)
    print('训练词向量 ok')

    # 2 save_model()保存已经训练好词向量
    # 注意，该行代码执行耗时很长
    mymodel.save_model(path="./data/fil9.bin")
    print('保存词向量 ok')

    # 3 模型加载
    mymodel = fasttext.load_model(path='./data/fil9.bin')
    print('加载词向量 ok')

if __name__ == '__main__':
    dm_fasttext_train_save_load()