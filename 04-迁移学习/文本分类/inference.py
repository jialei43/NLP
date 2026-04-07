import torch
from model import clsModel
from data_process import get_dataloader, my_tokenizer,device

# 加载模型
model = clsModel().to(device=device)
weight = torch.load(r"train_model/my_model_class_2.bin")
model.load_state_dict(weight)
model.eval()

# 加载数据
dataloader_train, dataloader_test, dataloader_eval = get_dataloader()

# 遍历数据进行预测
correct = 0
total = 0
# 遍历数据
for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(dataloader_eval):
    # 模型推理
    with torch.no_grad():
        out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    # 获取预测结果
    out = out.argmax(-1)
    # 获取预测正确的样本数
    correct += (out == labels).sum()
    # 总样本数
    total += labels.size(0)
    # 每隔10个迭代计算准确率
    # if i % 10 == 0:
    print(correct / total, end=" ")
    # id-》word
    print(my_tokenizer.decode(input_ids[0], skip_special_tokens=True), end=" ")
    print(f'预测值: {out[0].item()},真实值:{labels[0].item()}' )
# 计算总准确率
acc = correct / total
print(acc)
