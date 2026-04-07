# 使用save_model保存模型
import fasttext

def dm01_train_model():
    # 在训练词向量过程中, 我们可以设定很多常用超参数来调节我们的模型效果, 如:
    # 无监督训练模式: 'skipgram' 或者 'cbow', 默认为'skipgram', 在实践中，skipgram模式在利用子词方面比cbow更好.
    # 词嵌入维度dim: 默认为100, 但随着语料库的增大, 词嵌入的维度往往也要更大.
    # 数据循环次数epoch: 默认为5, 但当你的数据集足够大, 可能不需要那么多次.
    # 学习率lr: 默认为0.05, 根据经验, 建议选择[0.01，1]范围内.
    # 使用的线程数thread: 默认为12个线程, 一般建议和你的cpu核数相同.
    # model = fasttext.train_unsupervised("data/fil9", model='skipgram', dim=100, epoch=3)

    # model.save_model("data/fil9.bin")

    # 使用fasttext.load_model加载模型
    model = fasttext.load_model("data/fil9.bin")
    print(model.get_word_vector("good"))
    print(model.get_nearest_neighbors("good"))

def dm02_test_model():
    # 加载模型
    model = fasttext.load_model("model/cc.zh.300.bin")

    # 查看前100个词汇(这里的词汇是广义的, 可以是中文符号或汉字))
    print(len(model.words))
    print(model.get_word_vector("北京"))
    print(model.get_nearest_neighbors("美丽"))

if __name__ == '__main__':
    dm02_test_model()