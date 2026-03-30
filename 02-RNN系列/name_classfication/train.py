# 模型训练
import os

import torch
from torch import nn

from model import RNNModel
from ModelType import ModelType
from data_preprcessing import categories_num, letters_num, data_loader
from tqdm import tqdm

# device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(hidden_size, num_layers, model_type):
    # 模型实例化
    print(f'开始训练模型:{model_type}')
    model = RNNModel(letters_num, hidden_size, num_layers, categories_num, model_type)
    model = model.to(device)
    model.train()
    # 数据集实例化
    dataload = data_loader()

    # 损失
    loss_fn = nn.NLLLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    # 定义轮次
    epochs = 1000
    # 统计 loss
    total_iter = 0
    total_loss = 0
    loss_avg = 0
    # 统计损失函数用于绘图
    loss_list = []
    # 统计准确率
    acc_list = []
    total_correct = 0
    total_num = 0
    acc_avg = 0
    # 最优模型
    best_acc = 0

    for epoch in range(epochs):
        for x, y in tqdm(dataload):

            x = x.to(device)
            y = y.to(device)
            current_batch_size = x.size(0)
            # 初始化h0 和 c0
            h0, c0 = model.init_hidden(current_batch_size)
            h0 = h0.to(device)
            if c0 is not None:
                c0 = c0.to(device)
            # 模型预测
            # out, hn, cn = model(x[0], h0)
            if model_type == ModelType.LSTM:
                out, hn, cn = model(x, (h0,c0))
            else:
                out, hn,_ = model(x, h0)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            total_iter += 1
            total_num += y.shape[0]
            total_correct += torch.sum(torch.argmax(out, dim=-1) == y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 用于绘图
            if total_iter % 100 == 0:
                # 损失函数
                loss_list.append(total_loss / total_iter)
                # 准确率
                acc_list.append(total_correct.item() / total_num)

        acc_avg = total_correct / total_num
        loss_avg = total_loss / total_num
        # 3.打印训练结果
        print(f"epoch: {epoch + 1}/{epochs} | "
              f"train_loss: {loss_avg:.4f}, "
              f"train_acc: {acc_avg:.4f}"
              f"")
        if acc_avg > best_acc and acc_avg > 0.85:
            best_acc = acc_avg
            print(f"save model, best_acc: {best_acc:.4f}")
            os.makedirs(r"./model", exist_ok=True)
            if model_type == ModelType.LSTM:
                torch.save(model.state_dict(), r"./model/my_lstm.pth")
            elif model_type == ModelType.GRU:
                torch.save(model.state_dict(), r"./model/my_gru.pth")
            else :
                torch.save(model.state_dict(), r"./model/my_rnn.pth")


if __name__ == '__main__':
    hidden_size = 128
    num_layers = 1
    train_model(hidden_size, num_layers, model_type=ModelType.GRU)
