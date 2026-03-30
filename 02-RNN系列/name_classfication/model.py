import torch
from tensorflow.lite.python.conversion_metadata_schema_py_generated import ModelType
from torch import nn
from data_preprcessing import categories_num, letters_num, get_data, data_loader, NameClassDataSet
from ModelType import ModelType

# RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size,model_type):
        """
        初始化
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            num_layers: RNN堆叠的层数
            output_size: 输出维度
        """
        # 继承父类初始化
        super(RNNModel, self).__init__()
        # 词向量维度
        self.input_size = input_size
        # 隐藏层维度
        self.hidden_size = hidden_size
        # RNN堆叠的层数
        self.num_layers = num_layers
        # 输出维度
        self.output_size = output_size
        # 模型类型
        self.model_type = model_type
        # 定义RNN
        if model_type == ModelType.LSTM:
            # batch_first=true 此时输入 x 就必须是 [batch, seq_len, input_size]
            # 输出 out 也会变为 [batch, seq_len, hidden_size]
            # 本质：batch_first 只是改变 RNN 接受输入和返回输出张量的维度顺序，不改变计算逻辑。
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        elif model_type == ModelType.GRU:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        # 定义激活函数
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0,c0=None):
        # x: (seq_len, input_size)
        # hidden: (num_layers, batch, hidden_size)
        cn = None
        # 将x的维度变为 (seq_len, batch, input_size)
        # x = x.unsqueeze(dim=1)
        x = x.transpose_(0, 1)
        if self.model_type == ModelType.LSTM:
            out, (hn, cn) = self.rnn(x, (h0,c0))
        else:
            out, hn = self.rnn(x, h0)
        # 输出维度: (seq_len, batch, hidden_size)-> (batch, hidden_size)
        # 获取最后一个时间步的输出结果，通过索引切片来获取最后一步的输出结果
        out = out[-1]
        # 全连接层 映射到输出维度 (batch, hidden_size)-> (batch, output_size)
        out = self.fc(out)
        # 激活函数
        out = self.logsoftmax(out)
        # 返回
        return out, hn,cn
    def init_hidden(self, batch_size=1):
        """
        初始化隐藏层
        Args:
            batch_size: 批次大小
        Returns:
            初始化后的隐藏层
        """
        # 创建一个全0的矩阵
        hidden=c= torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden,c

def val_model(model_type):
    """

    :param model_type:
    :return:
    """
    hidden_size = 32
    num_layers = 1
    rnn = RNNModel(letters_num, hidden_size, num_layers, categories_num,model_type)
    x_tensor, y_tensor = get_data()
    dataset = NameClassDataSet(x_tensor, y_tensor)
    x_tensor, y_tensor = dataset.__getitem__(0)
    hn,cn = rnn.init_hidden()
    if model_type == ModelType.LSTM:
        out, hm, cn = rnn(x_tensor, hn,cn)
    else:
        out, hm,_ = rnn(x_tensor, hn)

    # print(out)
    # print(hn)
    # print(out.shape)
    # print(hn.shape)
    print(f'{model_type}的输出内容为:{out}')
    print(f'{model_type}的输出维度为：{out.shape}')
    for i in range(x_tensor.shape[0]):
        # rnnmodel 输入数据 x_tensor 的维度为 (seq_len, input_size)
        # x_tensor[index] 的维度为 (input_size) -> (1, input_size)
        input = x_tensor[i].unsqueeze(dim=0)
        if model_type == ModelType.LSTM:
            out, hn, cn = rnn(input, hn,cn)
        else:
            out, hn,_ = rnn(input, hn)
    print(f'{model_type}的输出内容为:{out}')
    print(f'{model_type}的输出维度为：{out.shape}')


if __name__ == '__main__':
    # 验证模型RNN
    # val_model(model_type=ModelType.RNN)
    # 验证模型LSTM
    # val_model(model_type=ModelType.LSTM)
    # 验证模型GRU
    val_model(model_type=ModelType.GRU)

