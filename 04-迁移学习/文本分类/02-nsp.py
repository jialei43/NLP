import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import random
import time

# 加载字典和分词工具
my_tokenizer = BertTokenizer.from_pretrained('../model/bert-base-chinese')

# 加载预训练模型
my_model_pretrained = BertModel.from_pretrained('../model/bert-base-chinese')

# 查看预训练模型的输出维度
hidden_size = my_model_pretrained.config.hidden_size
print('hidden_size--->', hidden_size)  # 768

class MyDataset(Dataset):
    def __init__(self, data_csv_files):
        # 生成数据源dataset对象
        my_dataset_temp = load_dataset('csv', data_files=data_csv_files, split="train")
        # print('my_dataset_temp--->', my_dataset_temp)

        # 按照条件过滤数据源对象
        self.my_dataset = my_dataset_temp.filter(lambda x: len(x['text']) > 44)
        # print('self.my_dataset--->', self.my_dataset)
        # print('self.my_dataset[0:3]-->', self.my_dataset[0:3])

        self.length = len(self.my_dataset)

    def __len__(self):
        return self.length # 7472

    def __getitem__(self, index):
        # 1是下一句话 0不是下一句
        label = 1
        text = self.my_dataset[index]['text']
        sentence1 = text[0:22]
        sentence2 = text[22:44]

        # 产生负样本: 随机产生0和1 一般概率选中0, 替换为无关的一句话
        if random.randint(0, 1) == 0:
            j = random.randint(0, self.length - 1)
            sentence2 = self.my_dataset[j]['text'][22:44]
            label = 0

        # 返回两句话 和两句话之间的关系
        return sentence1, sentence2, label


# 数据集处理自定义函数
def collate_fn3(data):
    # data -> [('今天去给笔记本贴膜，突然发现网卡有问题，给京', '很仔细地测试过这个上网本了。对于价格不超过2', 0), ...]
    sents = [i[:2] for i in data]
    labels = [i[2] for i in data]

    # 文本数值化
    data = my_tokenizer(text=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=50, # 44+cls+sep+sep+other=44+3=47
                                   return_tensors='pt')

    # input_ids 编码之后的数字
    # attention_mask 是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    # 注意labels不要忘记需要转成tensor 1维数组
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels

def dm01_test_dataset():

    data_files = '../data/train.csv'
    my_dataset = MyDataset(data_files)
    print('my_dataset-->', my_dataset, len(my_dataset))
    # print(my_dataset[3])

    # 通过dataloader进行迭代
    my_dataloader = DataLoader(my_dataset,
                               batch_size=8,
                               collate_fn=collate_fn3,
                               shuffle=True,
                               drop_last=True)
    print('my_dataloader--->', my_dataloader)
    for (input_ids, attention_mask, token_type_ids, labels) in my_dataloader:
        print(my_tokenizer.decode(input_ids[0])) # 打印每个批次的第1句话
        print(labels[0]) # 打印每个批次的第1句话
        print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)
        break

# 定义下游任务模型NSP
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):

        # 预训练模型不训练
        with torch.no_grad():
            out = my_model_pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        # 下游任务模型训练 数据经过全连接层
        # out = self.fc(out.last_hidden_state[:, 0])
        output = self.fc(out.pooler_output)

        return output
# NSP模型输入和输出测试
def dm02_test_mymodel():

    data_files = '../data/train.csv'
    my_dataset = MyDataset(data_files)
    print('my_dataset-->', my_dataset, len(my_dataset))

    # 通过dataloader进行迭代
    my_dataloader = DataLoader(my_dataset, batch_size=4, collate_fn=collate_fn3, shuffle=True, drop_last=True)
    print('my_dataloader--->', my_dataloader)

    # 不训练,不需要计算梯度
    for param in my_model_pretrained.parameters():
        param.requires_grad_(False)

    # 实例化下游任务模型
    my_model = MyModel()

    # 给模型喂数据
    for (input_ids, attention_mask, token_type_ids, labels) in my_dataloader:
        print(my_tokenizer.decode(input_ids[0]))
        print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

        # 给模型喂数据 [8,768] ---> [8,2] nsp任务是二分类
        out = my_model(input_ids, attention_mask, token_type_ids)
        print('out--->', out.shape, out)
        break

# 模型训练NSP
def dm03_train_model():
    data_files = '../data/train.csv'
    my_dataset = MyDataset(data_files)
    # print('my_dataset-->', my_dataset, len(my_dataset))
    # print(my_dataset[3])

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
    epochs = 10

    # 设置模型为训练模型
    my_model.train()

    # 外层for循环 控制轮数
    for epoch_idx in range(epochs):

        # 实例化数据迭代器对象my_dataloader
        my_dataloader = DataLoader(my_dataset,
                                   batch_size=4,
                                   collate_fn=collate_fn3,
                                   shuffle=True,
                                   drop_last=True)
        starttime = int(time.time())

        # 内存for循环 控制迭代次数
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader):
            # 给模型喂数据 [8,50] --> [8,2]
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
                out = my_out.argmax(dim=1)  # [8,2] --> (8,)
                acc = (out == labels).sum().item() / len(labels)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d' \
                      % (epoch_idx, i, my_loss.item(), acc, int(time.time()) - starttime))
                break

        # 每个轮次保存模型
    torch.save(my_model.state_dict(), 'my_model_nsp.pt')

# 模型测试
def dm04_evaluate_model():

    # 实例化数据源对象my_dataset_test
    data_files = '../data/test.csv'
    my_dataset = MyDataset(data_files)

    # 实例化下游任务模型my_model
    path = 'my_model_nsp.pt'
    my_model = MyModel()
    my_model.load_state_dict(torch.load(path))
    print('my_model-->', my_model)

    # 设置下游任务模型为评估模式
    my_model.eval()

    # 设置评估参数
    correct = 0
    total = 0

    # 实例化化dataloader
    my_loader_test = DataLoader(my_dataset,
                                batch_size=8,
                                collate_fn=collate_fn3,
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

        if i % 20 == 0:
            print(correct / total)
    print('全部样本',correct / total)

if __name__ == '__main__':
    # dm01_test_dataset()
    # dm02_test_mymodel()
    # dm03_train_model()
    dm04_evaluate_model()