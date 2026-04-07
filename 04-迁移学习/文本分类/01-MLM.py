import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import time

# 加载字典和分词工具
my_tokenizer = BertTokenizer.from_pretrained('../model/bert-base-chinese')

# 加载预训练模型
my_model_pretrained = BertModel.from_pretrained('../model/bert-base-chinese')

# 查看预训练模型的输出维度
hidden_size = my_model_pretrained.config.hidden_size

def dm_file2dataset():
    # 获取训练数据集
    train_dataset_tmp = load_dataset('csv', data_files='../data/train.csv', split='train')
    print('train_dataset_tmp--->', train_dataset_tmp)
    print('train_dataset_tmp[0]--->', train_dataset_tmp[0])
    # 过滤掉样本的评论内容长度小于等于32的样本
    # x->{'label':0, 'text':xxxx}
    my_train_dataset = train_dataset_tmp.filter(lambda x: len(x['text']) > 32)
    print('my_train_dataset--->', my_train_dataset)

    # 获取测试数据集
    test_dataset_tmp = load_dataset('csv', data_files='../data/test.csv', split='train')
    my_test_dataset = test_dataset_tmp.filter(lambda x: len(x['text']) > 32)

    return my_train_dataset, my_test_dataset

def collate_fn2(data):
    sents = [i['text'] for i in data]
    """
    这是一个非常典型的深度学习库在版本迭代中“规范化”参数名称的过程。
    1. batch_text_or_text_pairs 的含义从字面意思来看，它是 “批量的文本或文本对”。在 NLP 的预训练模型（如 BERT）中，Tokenizer 
    通常需要处理两种情况：单句输入： [CLS] 文本 [SEP] (用于分类任务)。双句输入： [CLS] 文本A [SEP] 文本B [SEP] (用于句子对关系判断，如 NLI 或 QA)。
    在旧版本的 Transformers 源码中，batch_text_or_text_pairs 是 Tokenizer 内部 __call__ 方法的第一个位置参数的名字。
    它存在的目的是告诉 Tokenizer：你可以传一个列表进来（Batch），列表里可以是单独的字符串，也可以是元组（文本对）。
    2. 改成 text 会有什么影响？结论：没有任何负面影响，反而会让你的代码更具兼容性和可读性。以下是详细的影响分析：解决报错（最直接的影响）： 
    在新版 transformers 中，内部逻辑会显式检查 text 关键字是否被赋值。如果你写 batch_text_or_text_pairs=sents，新版的校验逻辑可能无法将其正确映射到 text 逻辑块中，导致它认为你“什么都没传”，
    从而报出你刚才看到的 ValueError。
    逻辑完全等价： 在 Tokenizer 的实现中，text 参数本身就支持多种格式。无论是 text="一段话"，还是 text=["第一段", "第二段"]，甚至是 text=[("句A", "句B")]，它都能通过内部判断自动处理。提高鲁棒性： 
    使用 text 是 Hugging Face 官方推荐的标准写法。这意味着无论后续库如何更新内部变量名（比如把内部变量改成 first_input 之类的），只要 text 这个公共 API 接口不变，你的代码就不会崩。
    3. 为什么新版本不让用旧名字了？这是为了适配 Seq2Seq（序列到序列） 任务（如翻译、摘要）。新版 Tokenizer 需要区分：输入端文本 (text)：比如一段英文。目标端文本 (text_target)：比如对应的中文翻译。为了保证逻辑严密，
    代码现在会强制要求你明确指定你在喂哪一部分数据。旧的参数名 batch_text_or_text_pairs 太过偏向底层实现描述，不符合现在这种“双端输入”的架构逻辑，因此被边缘化甚至废弃了。
    总结你完全可以放心大胆地改为 text=sents。如果你传的是单句列表： [S1, S2, S3] $\rightarrow$ 正常处理。如果你传的是双句列表： [(A1, B1), (A2, B2)] $\rightarrow$ 依然正常处理。一句话建议：
     永远优先使用官方文档中定义的参数名（如 text, text_pair），避免使用库内部的实现变量名。
    """
    # 文本数值化
    data = my_tokenizer(text=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=32,
                                   return_tensors='pt')

    # input_ids 编码之后的数字
    # attention_mask 是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    # 取出每批的8个句子 在第16个位置clone出来 做真实标签
    labels = input_ids[:, 16].clone()

    # 将第16个词替换成[MASK]的下标值
    # 获取[MASK]字符
    # print(my_tokenizer.mask_token)
    # 获取[MASK]字符的下标
    # print(my_tokenizer.mask_token_id)
    input_ids[:, 16] = my_tokenizer.get_vocab()[my_tokenizer.mask_token]


    # tmpa = input_ids[:, 16]
    # print('tmpa--->', tmpa, tmpa.shape)       # torch.Size([8])
    # print('labels-->', labels.shape, labels)  # torch.Size([8])

    return input_ids, attention_mask, token_type_ids, labels

# 数据源 数据迭代器 测试
def dm01_test_dataset():

    # 生成数据源dataset对象
    dataset_train_tmp = load_dataset('csv', data_files='../data/train.csv', split="train")
    # print('dataset_train_tmp--->', dataset_train_tmp)

    # 按照条件过滤数据源对象
    my_dataset_train = dataset_train_tmp.filter(lambda x: len(x['text']) > 32)
    # print('my_dataset_train--->', my_dataset_train)
    # print('my_dataset_train[0:3]-->', my_dataset_train[0:3])

    # 通过dataloader进行迭代
    my_dataloader = DataLoader(my_dataset_train,
                               batch_size=8,
                               collate_fn=collate_fn2,
                               shuffle=True,
                               drop_last=True)
    print('my_dataloader--->', my_dataloader)

    # 调整数据迭代器对象数据返回格式
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader):

        print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

        print('\n第1句mask的信息')
        print(my_tokenizer.decode(input_ids[0]))
        print(my_tokenizer.decode(labels[0]))
        print(my_tokenizer.decode(labels[0]))

        print('\n第2句mask的信息')
        print(my_tokenizer.decode(input_ids[1]))
        print(my_tokenizer.decode(labels[1]))
        break


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层
        self.fc = nn.Linear(768, my_tokenizer.vocab_size, bias=False)
        # 设置全连接层偏置为零
        self.fc.bias = nn.Parameter(torch.zeros(my_tokenizer.vocab_size))

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 预训练模型不进行训练
        with torch.no_grad():
            out = my_model_pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        # 下游任务进行训练 形状[8,768] ---> [8, 21128]
        out = self.fc(out.last_hidden_state[:, 16])

        # 返回
        return out

# 模型输入和输出测试
def dm02_test_mymodel():
    # 生成数据源dataset对象
    dataset_train_tmp = load_dataset('csv', data_files='../data/train.csv', split="train")
    # print('dataset_train_tmp--->', dataset_train_tmp)

    # 按照条件过滤数据源对象
    my_dataset_train = dataset_train_tmp.filter(lambda x: len(x['text']) > 32)
    # print('my_dataset_train--->', my_dataset_train)
    # print('my_dataset_train[0:3]-->', my_dataset_train[0:3])

    # 通过dataloader进行迭代
    my_dataloader = DataLoader(my_dataset_train,
                               batch_size=8,
                               collate_fn=collate_fn2,
                               shuffle=True,
                               drop_last=True)
    print('my_dataloader--->', my_dataloader)

    # 不训练,不需要计算梯度
    for param in my_model_pretrained.parameters():
        param.requires_grad_(False)

    # 实例化下游任务模型
    mymodel = MyModel()

    # 调整数据迭代器对象数据返回格式
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader):

        print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

        print('\n第1句mask的信息')
        print(my_tokenizer.decode(input_ids[0]))
        print(my_tokenizer.decode(labels[0]))

        print('\n第2句mask的信息')
        print(my_tokenizer.decode(input_ids[1]))
        print(my_tokenizer.decode(labels[1]))

        # 给模型喂数据 [8,768] ---> [8,21128] 填空就是分类 21128个单词中找一个单词
        myout = mymodel(input_ids, attention_mask, token_type_ids)
        print('myout--->', myout.shape, myout)
        break


# 模型训练 - 填空
def dm03_train_model():

    # 实例化数据源对象my_dataset_train
    dataset_train_tmp = load_dataset('csv', data_files='../data/train.csv', split="train")
    my_dataset_train = dataset_train_tmp.filter(lambda x: len(x['text']) > 32)
    print('my_dataset_train--->', my_dataset_train)

    # 实例化下游任务模型my_model
    my_model = MyModel()

    # 实例化优化器my_optimizer
    my_optimizer = AdamW(my_model.parameters(), lr=5e-4)

    # 实例化损失函数my_criterion
    my_criterion = CrossEntropyLoss()

    # 不训练预训练模型 只让预训练模型计算数据特征 不需要计算梯度
    for param in my_model_pretrained.parameters():
        param.requires_grad_(False)

    # 设置训练参数
    epochs = 100

    # 设置模型为训练模型
    my_model.train()

    # 外层for循环 控制轮数
    for epoch_idx in range(epochs):

        # 实例化数据迭代器对象my_dataloader
        my_dataloader = torch.utils.data.DataLoader(my_dataset_train,
                                                    batch_size=8,
                                                    collate_fn=collate_fn2,
                                                    shuffle=True,
                                                    drop_last=True)
        starttime = int(time.time())
        # 内层for循环 控制迭代次数
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader, start=1):
            # 给模型喂数据 [8,32] --> [8,21128]
            my_out = my_model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

            # 计算损失
            my_loss = my_criterion(my_out, labels)

            # 梯度清零
            my_optimizer.zero_grad()

            # 反向传播
            my_loss.backward()

            # 梯度更新
            my_optimizer.step()

            # 每5次迭代 算一下准确率
            if i % 20 == 0:
                out = my_out.argmax(dim=1) # [8,21128] --> (8,)
                acc = (out == labels).sum().item() / len(labels)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d' \
                      %(epoch_idx, i, my_loss.item(), acc, int(time.time())-starttime))
                break

        # 每个轮次保存模型
    torch.save(my_model.state_dict(), 'my_model_mask.pt' )

# 模型测试：填空
def dm04_evaluate_model():

    # 实例化数据源对象my_dataset_test
    print('\n加载测试集')
    my_dataset_tmp = load_dataset('csv', data_files='../data/test.csv', split='train')
    my_dataset_test = my_dataset_tmp.filter(lambda x: len(x['text']) > 32)
    print('my_dataset_test--->', my_dataset_test)
    # print(my_dataset_test[0:3])

    # 实例化下游任务模型my_model
    path = 'my_model_mask.pt'
    my_model = MyModel()
    my_model.load_state_dict(torch.load(path))
    print('my_model-->', my_model)

    # 设置下游任务模型为评估模式
    my_model.eval()

    # 设置评估参数
    correct = 0
    total = 0

    # 实例化化dataloader
    my_loader_test = DataLoader(my_dataset_test,
                                batch_size=8,
                                collate_fn=collate_fn2,
                                shuffle=True,
                                drop_last=True)

    # 给模型送数据 测试预测结果
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(my_loader_test):
        with torch.no_grad():
            my_out = my_model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = my_out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

        if i % 25 == 0:
            print(i+1, my_tokenizer.decode(input_ids[0]))
            print('预测值:', my_tokenizer.decode(out[0]), '\t真实值:', my_tokenizer.decode(labels[0]))
            print(correct / total)
    print(correct / total)



if __name__ == '__main__':
    # my_train_dataset, my_test_dataset= dm_file2dataset()
    # dm01_test_dataset()
    # dm02_test_mymodel()
    # dm03_train_model()
    dm04_evaluate_model()
