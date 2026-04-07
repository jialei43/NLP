import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 加载字典和分词工具 实例化分词工具
my_tokenizer = BertTokenizer.from_pretrained('../model/bert-base-chinese')

# 加载预训练模型 实例化预训练模型
my_model_pretrained = BertModel.from_pretrained('../model/bert-base-chinese')

# 查看预训练模型的输出维度
hidden_size = my_model_pretrained.config.hidden_size
print('hidden_size--->', hidden_size)  # 768
def get_dataset():
    """
    在使用 Hugging Face 的 datasets 库时，split 参数是一个非常实用且灵活的工具
    简单来说，当你在 load_dataset 中使用 split='train' 时，它的核心作用是：决定返回的数据结构，并指定加载数据集的哪一部分
    1、改变返回值的类型 (最直接的影响)
        这是新手最容易混淆的地方。根据是否设置 split，API 返回的对象完全不同：

        不设置 split 时： 返回一个 DatasetDict 对象（类似于 Python 字典）。

        即便你的文件夹里只有一个 CSV 文件，它也会默认包装成 {"train": Dataset}。

        你需要通过 dataset_train['train'] 才能访问数据。

        设置 split='train' 时： 直接返回一个 Dataset 对象。

        你可以直接进行 dataset_train[0] 或 dataset_train.map(...) 操作，少了一层索引。
    2. 指定加载的文件范围split 参数可以配合 data_files 来精细化控制加载哪些数据。
        用法示例实际效果split='train'寻找并加载被标记为训练集的数据（在你的例子中就是 train.csv）。
        split='test'如果 data_files 映射了测试集，则只加载测试部分。

        from datasets import load_dataset

        # 定义文件映射
        data_files = {
            "train": ["../data/train_part1.csv", "../data/train_part2.csv"],
            "test": "../data/test.csv",
            "validation": "../data/valid.csv"
        }

        # 通过 split 参数加载指定部分
        # 这会直接返回 Dataset 对象，包含 part1 和 part2 的合并数据
        train_dataset = load_dataset('csv', data_files=data_files, split='train')

        # 如果不传 split，则返回 DatasetDict，包含所有定义的 key
        # all_datasets = load_dataset('csv', data_files=data_files)

    3. 实现数据的自动切分 (切片功能)
        这是 split 参数最强大的地方。如果你只有一个巨大的 CSV 文件，但想在加载时直接分出训练集和验证集，可以像切片数组一样操作：

        加载前 10% 的数据：
        split='train[:10%]'

        加载 10% 到 20% 之间的数据：
        split='train[10%:20%]'

        混合切分：
        你可以利用这个参数在没有预先准备验证集的情况下，直接从 train.csv 中切出一部分作为验证集：

        Python
        train_data = load_dataset('csv', data_files="train.csv", split='train[:80%]')
        val_data = load_dataset('csv', data_files="train.csv", split='train[80%:]')

    :return:
    """
    dataset_train = load_dataset('csv', data_files="../data/train.csv", split='train')
    dataset_test = load_dataset('csv', data_files="../data/test.csv", split='train')
    dataset_val = load_dataset('csv', data_files="../data/validation.csv", split='train')
    return dataset_train, dataset_test, dataset_val

def collate_fn(datas):

    texts = [data["text"] for data in datas]
    labels = [data["label"] for data in datas]
    tokens = my_tokenizer(texts,padding='max_length',max_length=200,truncation=True,return_tensors='pt')
    # input_ids 编码之后的数字     含义：它是文本经过分词（Tokenization）并映射到词表（Vocabulary）后的索引数字序列
    # attention_mask 是补零的位置是0,其他位置是1
    """
    含义：一个与 input_ids 形状相同的 0/1 序列。1 表示该位置是真实的 Token，0 表示该位置是补全的填充符号（Padding）。
    作用：由于模型处理的数据必须是固定长度（如你代码中的 max_length=200），较短的句子会被补 [PAD]。
    attention_mask 告诉模型：“只计算 1 处的信息，忽略 0 处的填充，不要让无效的填充影响语义理解。”
    """
    # token_type_ids 是编码的数字对应的token类型
    """
    含义：又称 Segment Embeddings。用于区分两个不同的句子。0 代表第一句，1 代表第二句。

    作用：帮助模型理解句子之间的关系。
    """
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    token_type_ids = tokens['token_type_ids'].to(device)

    # 注意labels不要忘记需要转成tensor 1维数组
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, token_type_ids, labels
    # tokens['at']

def get_dataloader():
    #获取数据集
    dataset_train, dataset_val, dataset_test = get_dataset()
    # print(len(dataset_train), len(dataset_val), len(dataset_test))
    # print(dataset_val[0:5])
    # 获取batch_size样本
    dataloader_train = DataLoader(dataset=dataset_train, collate_fn=collate_fn, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, collate_fn=collate_fn, batch_size=128, shuffle=True)
    dataloader_val = DataLoader(dataset=dataset_val, collate_fn=collate_fn, batch_size=128, shuffle=True)
    return dataloader_train, dataloader_test, dataloader_val

if __name__ == '__main__':
    dataset_train, dataset_test, dataset_val = get_dataset()
    print(dataset_train)
    print(dataset_test)
    print(dataset_val)
    print(collate_fn(dataset_train))
    # print(len(dataset_train), len(dataset_val), len(dataset_test))
    # print(dataset_val[0:5])
    dataloader = DataLoader(dataset=dataset_train, collate_fn=collate_fn, batch_size=4, shuffle=True)
    for input_ids, token_type_ids, attention_mask, labels in dataloader:
        print(input_ids, token_type_ids, attention_mask, labels)
        break
