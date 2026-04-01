import os
import random
import time

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from data_processing import data_loader, SOS_token, device, EOS_token
from model import EncoderRNN,AttentionDecoderRNN
from torch import optim,nn
# 模型训练参数
mylr = 1e-4
epochs = 2
# 设置teacher_forcing比率为0.5
teacher_forcing_ratio = 0.5
print_interval_num = 1600
plot_interval_num = 128
def train():
    # 加载数据
    dataloader ,eng_word2idx, eng_idx2word, eng_word_num, fre_word2idx, fre_idx2word, fre_word_num = data_loader()

    # 模型实例化
    encoderRnn = EncoderRNN(eng_vocab_size=eng_word_num, hidden_size=32)
    atten_decoderRnn = AttentionDecoderRNN(fre_vocab_size=fre_word_num, hidden_size=32)
    encoderRnn.to(device)
    atten_decoderRnn.to(device)

    # 优化器
    encoder_optimzer = optim.Adam(params=encoderRnn.parameters(),lr=0.001,weight_decay=0.0001,betas=(0.9,0.999))
    atten_decoder_optimzer = optim.Adam(params=atten_decoderRnn.parameters(),lr=0.001,weight_decay=0.0001,betas=(0.9,0.999))

    # 损失函数
    nll_loss = nn.NLLLoss(ignore_index=2)

    # 统计参数初始化
    plot_loss_list=[]
    # 最优损失
    best_loss = 0
    # 训练
    for epoch in range(1,epochs+1):
        # 初始化打印和绘图用的损失累加器，记录开始时间
        print_loss_total,epoch_loss_total, plot_loss_total = 0.0, 0.0,0.0
        # 开始时间
        starttime = time.time()
        # 迭代次数
        iter_num = 0
        for x,y in tqdm(dataloader):
            x,y = x.to(device),y.to(device)
            my_loss = train_iter(x,y,encoderRnn,atten_decoderRnn,encoder_optimzer,atten_decoder_optimzer, nll_loss)
            # 累加损失用于后续统计
            print_loss_total += my_loss
            epoch_loss_total += my_loss
            plot_loss_total += my_loss
            iter_num += x.size(0)
            # 打印
            # 每隔一定迭代次数打印训练进度和平均损失-每隔1000次
            if iter_num % print_interval_num == 0:
                # 1000迭代的平均损失
                print_loss_avg = print_loss_total / print_interval_num
                # 将总损失归零
                print_loss_total = 0.0
                print(f'轮次:{epoch}/{epochs} | 迭代次数:{iter_num} | 训练损失:{print_loss_avg:.4f} | 耗时:{time.time()-starttime:.4f}s')
            # 每隔一定迭代次数记录损失用于绘图-每隔100次
            if iter_num % plot_interval_num == 0:
                # 100迭代平均损失
                plot_loss_avg = plot_loss_total / plot_interval_num
                plot_loss_list.append(plot_loss_avg)
                plot_loss_total = 0.0
        # 每个轮次的平均损失
        epoch_loss = epoch_loss_total / iter_num
        print(f'轮次:{epoch}/{epochs} | 训练损失:{epoch_loss:.4f} | 最优损失:{best_loss:.4f} | 耗时:{time.time()-starttime:.4f}s')
        # if epoch_loss < best_loss and best_loss != 0:
        # 保存模型
        os.makedirs(r'./model',exist_ok=True)
        torch.save(encoderRnn.state_dict(), r'./model/encoderRnn.pth')
        torch.save(atten_decoderRnn.state_dict(), r'./model/atten_decoderRnn.pth')
        best_loss = epoch_loss
        print(f'保存模型成功, | 轮次:{epoch}/{epochs} | 当前损失:{epoch_loss:.4f}')
    # 训练完成后绘制损失曲线并保存图像
    plt.figure()
    plt.plot(plot_loss_list)
    os.makedirs(r'./img', exist_ok=True)
    plt.savefig('img/s2sq_loss.png')
    plt.show()
    return plot_loss_list



def train_iter(x,y,encoderRnn,atten_decoderRnn,encoder_optimzer,atten_decoder_optimzer, nll_loss):
    # 模型模式
    encoderRnn.train()
    atten_decoderRnn.train()

    batch_size = x.size(0)
    target_length = y.size(1)

    # 初始化隐藏状态
    h0 = encoderRnn.initHidden(batch_size)
    # encoder模型训练
    encoder_output, encode_hidden = encoderRnn(x,h0)
    decoder_hidden = encode_hidden
    # 初始化输入数据q
    decoder_input = torch.tensor([[SOS_token]]*batch_size,device=device,dtype=torch.long)
    decoder_end = torch.tensor([[SOS_token]] * batch_size, device=device, dtype=torch.long)
    # 遍历目标句子长度
    loss_seq = 0.0
    teacher_forcing_ratio = 0.2
    teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    for i in range(target_length):
        # 取当前时间步的结果，保持 batch 维度
        # y[:, i]：y是当前Batch的目标语言张量，形状为[batch_size, seq_len]。
        # •: 表示选取所有的Batch（即所有8个样本）。
        # •    i表示选取当前时间步（第 $i$ 个单词）的索引。
        # •    结果形状：由[8, 12]降维成[8]，这是一个一维向量。
        # .unsqueeze(1)：
        # •    这个方法在索引为1的位置增加一个维度。
        # •    结果形状：由[8]变为[8, 1]。
        # decoder_input = y[:, i].unsqueeze(1)
        # 解码器前向传播
        # out 形状: [batch_size, fre_vocab_size]
        out, decoder_hidden, atten_weight = atten_decoderRnn(decoder_input, decoder_hidden, encoder_output)

        # --- 正确的 Loss 计算 ---
        # target 应该是当前步所有样本的真实单词索引: [batch_size]
        target_y = y[:, i]

        # nll_loss(预测概率分布 [8, vocab], 真实标签 [8])
        loss = nll_loss(out, target_y)
        loss_seq += loss
        # -----------------------

        # 5. 确定下一个时间步的输入
        if teacher_forcing:
            # Teacher Forcing: 下一次输入是真实的单词 [batch_size, 1]
            decoder_input = target_y.unsqueeze(1)
        else:
            # 自回归: 下一次输入是模型刚才预测出的单词 [batch_size, 1]
            topv, topi = out.topk(1)
            # 2. 准备下一轮的输入
            # .detach(): 将该张量从当前计算图中分离。这样在计算下一时刻的梯度时，
            # 不会一直追溯到这一时刻，防止梯度爆炸或显存溢出。
            decoder_input = topi.detach()

            # 检查 Batch 中每一个样本是否为 EOS (假设 EOS_token 为 1)
            # eos_batches 是一个布尔张量，例如 [True, False, True, ...]
            eos_batches = (decoder_input.squeeze(1) == EOS_token)
            # 如果你希望【只要有一个人没写完，就继续写】，
            # 那么只有当【所有人】都遇到 EOS 时才 break
            if eos_batches.all():
                break

        # 反向传播
        encoder_optimzer.zero_grad()
        atten_decoder_optimzer.zero_grad()
        loss_seq.backward()
        encoder_optimzer.step()
        atten_decoder_optimzer.step()

        # 返回样本的平均损失用于统计
        return loss_seq.item() / target_length


if __name__ == '__main__':
    train()