from torch import nn
import torch
import copy
from encoder import SublayerConnection, clones, MultiHeadedAttention, PositionwiseFeedForward, EncoderLayer, Encoder, \
    LayerNorm
from input import EmbeddingLayer, PositionalEncoding

class DecoderLayer(nn.Module):
    """
    解码层
    """
    def __init__(self, size, attn, corss_att,feed_forward, dropout=0.1):
        super().__init__()
        self.size = size
        self.self_attn = attn
        self.cross_attn = corss_att
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, encoder_output, src_mask, trg_mask):
        """
        前向传播
        Args:
            x: 输入
            encoder_output: 编码器的输出
            src_mask: 源序列的掩码
            trg_mask: 目标序列的掩码

        Returns:
            输出
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    """

    """
    def __init__(self, layer, N):
        super().__init__()
        # 层
        self.layers = clones(layer, N)
        # 归一化
        self.norm = LayerNorm(layer.size)
    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

if __name__ == '__main__':
    vocab = 1000
    d_model = 512
    dropout_p = 0.1
    max_len = 60
    head = 8
    my_embeddings = EmbeddingLayer(vocab, d_model)
    my_pe = PositionalEncoding(d_model, dropout_p, max_len)
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    x = my_embeddings(x)
    pe_result = my_pe(x)
    mask = torch.tril(torch.ones((8, 4, 4))).type(torch.uint8)
    c = copy.deepcopy
    # 编码器
    attn = MultiHeadedAttention(h=8, d_model=d_model, dropout=dropout_p)
    feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=2048, dropout_p=dropout_p)
    encoder_layer = EncoderLayer(size=d_model, self_attn=c(attn), feed_forward=c(feed_forward))
    encode = Encoder(encoder_layer, 4)
    encoder_result = encode(pe_result, mask)
    print('encoder_result.shape--->', encoder_result.shape)
    # 解码器
    decoder_layer = DecoderLayer(size=d_model, attn=c(attn), corss_att=c(attn), feed_forward=c(feed_forward))
    decoder = Decoder(decoder_layer, N=4)
    decoder_result = decoder(encoder_result, encoder_result, mask, mask)
    print('decoder_result.shape--->', decoder_result.shape)
