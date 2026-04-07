import os
import time

import torch
from torch import optim, nn

from data_process import get_dataloader,device
from model import clsModel

# 1.数据
dataloader_train,dataloader_test,dataloader_eval = get_dataloader()
# 2.模型
model = clsModel().to(device=device)
model.train()

# 3.优化器
optimzer = optim.AdamW(model.parameters(), lr=0.01)
# 损失
error = nn.CrossEntropyLoss()
epochs = 3


# 遍历epochs
for epoch_idx in range(epochs):
    total_loss = 0
    total_acc = 0
    total_num = 0
    # 遍历批次
    for i,(input_ids, token_type_ids, attention_mask, labels) in enumerate(dataloader_train):
        starttime = int(time.time())
        out = model(input_ids, token_type_ids, attention_mask)
        total_acc+= (out.argmax(dim=1) == labels).sum().item()
        # 计算损失
        my_loss = error(out, labels)
        total_loss += my_loss.item()
        total_num += len(labels)
        # 反向传播
        optimzer.zero_grad()
        my_loss.backward()
        optimzer.step()
        # print(my_loss.item())
        # 每5次迭代 算一下准确率
        # if i % 20 == 0:
    # out = out.argmax(dim=1)  # [8,2] --> (8,)
    acc_avg = total_acc / total_num
    loss_avg = total_loss / total_num

    print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%.4f' \
          % (epoch_idx, i, loss_avg, acc_avg, int(time.time()) - starttime))

        # 每个轮次保存模型
    os.makedirs('train_model', exist_ok=True)
    torch.save(model.state_dict(), 'train_model/my_model_class_%d.bin' % (epoch_idx + 1))








# 模型保存