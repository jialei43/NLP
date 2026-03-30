# 模型预测
import torch
from data_preprcessing import letters, letters_num, categories_num, categories
from model import RNNModel
from ModelType import ModelType
from data_preprcessing import categories_num, letters_num, data_loader
from train import device



# 将输入name转换成one-hot
# 7 57
def name2onehot(name):
    MAX_LENGTH = 10
    # # 全0
    # onehot = torch.zeros(len(name), letters_num)
    # # print(onehot.size())
    # # 遍历每个确定one-hot
    # for i, letter in enumerate(name):
    #     id = letters.find(letter)
    #     onehot[i][id] = 1
    # # print(onehot)
    # return onehot

    tensor_x = torch.zeros(MAX_LENGTH, letters_num, dtype=torch.float32)

    # 遍历姓名，如果姓名超过 MAX_LENGTH 则截断，不足则保留为0（即Padding）
    for i, letter in enumerate(name[:MAX_LENGTH]):
        if letter in letters:
            tensor_x[i, letters.index(letter)] = 1

    return tensor_x


# rnn预测
def model_predict(hidden_size, num_layers, model_type,name):
    onehot = name2onehot(name).to(device)
    onehot = onehot.unsqueeze(dim=1)
    # 参数与训练时一样
    model = RNNModel( letters_num, hidden_size, num_layers, categories_num, model_type)
    # 加载权重
    model_name = f'./model/my_{model_type}.pth'
    weights = torch.load(model_name)

    # 匹配权重
    model.load_state_dict(weights)
    # 验证模式
    model.eval()
    # 预测
    with torch.no_grad():
        current_batch_size = onehot.size(0)
        # 初始化h0 和 c0
        h0, c0 = model.init_hidden(current_batch_size)
        h0 = h0.to(device)
        if c0 is not None:
            c0 = c0.to(device)
        # 模型预测
        # out, hn, cn = model(x[0], h0)
        if model_type == ModelType.LSTM:
            out, hn, cn = model(onehot, (h0, c0))
        else:
            out, hn, _ = model(onehot, h0)
        # print(out)
        # 获取最大
        # id = torch.argmax(out,dim=-1)
        # print(categories[id])
        # topK个结果
        # 1、topk的功能
        # tensor.topk(k, dim, largest=True, sorted=True)会返回：前k个最大值（或者最小值，如果largest = False）对应的索引
        # 返回两个张量： 值和索引

        top_value, top_id = out.topk(k=3, dim=-1, largest=True)
        # print(top_value)
        # print(top_id[0])
        # 获取可能的国家
        top_nations = []
        for id in top_id[0]:
            top_nations.append(categories[id])
    # 返回
    return top_nations





if __name__ == '__main__':
    hidden_size = 128
    num_layers = 1
    model_list = [ModelType.RNN, ModelType.LSTM, ModelType.GRU]
    print(model_predict(hidden_size, num_layers, ModelType.GRU, 'Oppenheimer'))
    # for model_type in model_list:
    #     print(model_predict(hidden_size, num_layers, model_type, 'James'))

