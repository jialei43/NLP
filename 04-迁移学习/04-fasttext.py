# 导入fasttext
from tabnanny import verbose

import fasttext
def dm01_train_fasttext():
    # 使用fasttext的train_supervised方法进行文本分类模型的训练
    model = fasttext.train_supervised(input="data/cooking.pre.train", lr=0.5, epoch=50, wordNgrams=3, loss='ova')
    print(model.predict("Which baking dish is best to bake a banana bread ?",k=3))

    # 使用fasttext的test方法进行模型评估
    print(model.test("data/cooking.valid"))
    model.save_model('model/cooking.model')
    load_model = fasttext.load_model('model/cooking.model')
    print(load_model.test("data/cooking.valid"))

def dm02_auto_fasttext():
    # 使用fasttext的load_model方法进行模型加载
    # 手动调节和寻找超参数是非常困难的, 因为参数之间可能相关, 并且不同数据集需要的超参数也不同,
    # 因此可以使用fasttext的autotuneValidationFile参数进行自动超参数调优.
    # autotuneValidationFile参数需要指定验证数据集所在路径, 它将在验证集上使用随机搜索方法寻找可能最优的超参数.
    # 使用autotuneDuration参数可以控制随机搜索的时间, 默认是300s, 根据不同的需求, 我们可以延长或缩短时间.
    # 验证集路径'cooking.valid', 随机搜索600秒
    model = fasttext.train_supervised(input='data/cooking.pre.train', autotuneValidationFile='data/cooking.pre.valid',
                                      autotuneDuration=600,verbose=4)
    print(model.predict("Which baking dish is best to bake a banana bread ?",k=3))

    # 使用fasttext的test方法进行模型评估
    print(model.test("data/cooking_pre.valid"))
if __name__ == '__main__':
    dm01_train_fasttext()
    # dm02_auto_fasttext()