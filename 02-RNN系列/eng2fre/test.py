import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------- 配置 -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_LENGTH = 8  # 统一最大长度

data_path = '../data/eng-fra-v2.txt'  # 数据路径


# ----------------- 数据处理 -----------------
def normalize_string(s: str):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-z.!?]+", r" ", s)
    return s


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        pairs = [[normalize_string(s) for s in line.strip().split('\t')] for line in f]

    # 构建词表
    eng_word2idx = {"SOS": SOS_token, "EOS": EOS_token, "PAD": PAD_token}
    fre_word2idx = {"SOS": SOS_token, "EOS": EOS_token, "PAD": PAD_token}
    eng_idx = fre_idx = 3
    for pair in pairs:
        for idx, sentence in enumerate(pair):
            for word in sentence.split():
                if idx == 0:
                    if word not in eng_word2idx:
                        eng_word2idx[word] = eng_idx
                        eng_idx += 1
                else:
                    if word not in fre_word2idx:
                        fre_word2idx[word] = fre_idx
                        fre_idx += 1

    eng_idx2word = {v: k for k, v in eng_word2idx.items()}
    fre_idx2word = {v: k for k, v in fre_word2idx.items()}

    return pairs, eng_word2idx, eng_idx2word, fre_word2idx, fre_idx2word


class TranslationDataset(Dataset):
    def __init__(self, pairs, eng_word2idx, fre_word2idx):
        self.pairs = pairs
        self.eng_word2idx = eng_word2idx
        self.fre_word2idx = fre_word2idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eng, fre = self.pairs[idx]

        # 英文
        x = [self.eng_word2idx[w] for w in eng.split()]
        if len(x) < MAX_LENGTH:
            x += [PAD_token] * (MAX_LENGTH - len(x))
        else:
            x = x[:MAX_LENGTH]
        x = torch.tensor([SOS_token] + x + [EOS_token], dtype=torch.long, device=device)

        # 法文
        y = [self.fre_word2idx[w] for w in fre.split()]
        if len(y) < MAX_LENGTH:
            y += [PAD_token] * (MAX_LENGTH - len(y))
        else:
            y = y[:MAX_LENGTH]
        y = torch.tensor([SOS_token] + y + [EOS_token], dtype=torch.long, device=device)

        return x, y


def get_dataloader(batch_size=64):
    pairs, eng_word2idx, eng_idx2word, fre_word2idx, fre_idx2word = read_data(data_path)
    dataset = TranslationDataset(pairs, eng_word2idx, fre_word2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, eng_word2idx, eng_idx2word, fre_word2idx, fre_idx2word


# ----------------- 模型 -----------------
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.gru(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttentionDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_length=MAX_LENGTH + 1, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size, 1]
        embedded = self.embedding(input)  # [batch, 1, hidden]
        embedded = self.dropout(embedded)

        hidden_for_attn = hidden.permute(1, 0, 2)  # [batch, 1, hidden]
        attn_input = torch.cat((embedded, hidden_for_attn), dim=2)  # [batch,1,hidden*2]
        attn_weights = torch.softmax(self.attn(attn_input), dim=-1)  # [batch,1,max_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch,1,hidden]

        output = torch.cat((embedded, context), dim=2)
        output = torch.relu(self.attn_combine(output))
        output, hidden = self.gru(output, hidden)
        output = self.out(output.squeeze(1))
        output = self.logsoftmax(output)
        return output, hidden, attn_weights


# ----------------- 测试 -----------------
if __name__ == '__main__':
    dataloader, eng_word2idx, eng_idx2word, fre_word2idx, fre_idx2word = get_dataloader(batch_size=8)

    encoder = EncoderRNN(vocab_size=len(eng_word2idx), hidden_size=32).to(device)
    decoder = AttentionDecoderRNN(vocab_size=len(fre_word2idx), hidden_size=32, max_length=MAX_LENGTH + 1).to(device)

    for x, y in dataloader:
        batch_size = x.size(0)
        hidden = encoder.init_hidden(batch_size)
        encoder_outputs, hidden = encoder(x, hidden)

        # 解码器按时间步
        decoder_hidden = hidden
        for t in range(y.size(1)):
            decoder_input = y[:, t].unsqueeze(1)
            output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            print(f"Time step {t}: output shape {output.shape}, attn shape {attn_weights.shape}")
        break