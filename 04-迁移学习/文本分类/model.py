import torch
from torch import nn
from transformers import BertModel

from data_process import get_dataloader,device

#加载预测模型
pre_trained = BertModel.from_pretrained('../model/bert-base-chinese').to(device)
d_model = pre_trained.config.hidden_size
# 冻结参数，不更新
for param in pre_trained.parameters():
    param.requires_grad_(False)


# 构建模型
class clsModel(nn.Module):
    def __init__(self):
        super(clsModel, self).__init__()
        # 输出层定义
        self.fc = nn.Linear(in_features=d_model, out_features=2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 预训练模型前向传播  # 预训练模型不训练 只进行特征抽取 [8,500] ---> [8,768]
        with torch.no_grad():
            out = pre_trained(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # 输出层处理
        # 下游任务模型训练 数据经过全连接层 [8,768] --> [8,2]
        # out.last_hidden_state: 最后一层的隐藏状态张量
        # [:, 0] 选择的是序列的第一个token的隐藏状态
        # 通常这个token是特殊的[CLS]，该token被设计用于表示整个序列的语义。
        # BERT训练时，特别是文本分类任务，使用[CLS]的表示来作为整个句子的表示。
        # out = self.fc(out.last_hidden_state[:, 0])
        # pooler_output: 通过last_hidden_state[:, 0]拿到[CLS]向量表示后又经过linear层(形状不变)
        out = self.fc(out.pooler_output)
        return out


# 测试
if __name__ == '__main__':
    # 模型实例化
    model = clsModel().to(device=device)
    # 加载数据
    dataloader_train,dataloader_test,dataloader_eval= get_dataloader()
    # 数据处理
    for input_ids, token_type_ids, attention_mask, labels in dataloader_train:
        out = model.forward(input_ids, token_type_ids, attention_mask)
        print(out)
        print(out.shape)
        break
