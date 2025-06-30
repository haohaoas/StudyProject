
'''
- 作用：模型架构定义
- 利用nn.Transformer()，注意：
1. 接收的是位置编码与embedding层相加后的向量(可用之前写的transformer输入部分代码，注意位置编码转移设备)
2. 前向传播使需传入掩码张量，可用transformer内置的生成掩码张量方法
3. 还需一个输出线性层 (softmax可省列，因为后面的nn.CrossEntropyLoss()内部包含了softmax
4. 注意调参整体小一点
'''
import math

import torch
from torch import nn

class InputLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emd = Embeddings(vocab_size, d_model)  # embedding层
        self.pe = PositionEncoding(d_model)  # 位置编码层
        self.dropout = nn.Dropout(0.1)
    def forward(self, idxs):
        # idxs (batch,seq_len)
        # 注意切片及广播相加
        emb = self.emd(idxs)  # (batch,seq_len,d_model)
        pe = self.pe.positional_encoding[:, :emb.shape[1]].to(idxs.device)  # 注意位置编码转移设备
        # self.pe.positional_encoding # (1, max_seq_len,d_model) -切片> (1, seq_len,d_model)
        # 注意这里的切片是为了让位置编码和embedding的seq_len对齐
        return self.dropout(emb + pe)  # (batch,seq_len,d_model)

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        '''
        vocab_size: 词表大小
        d_model: 模型维度
        '''
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, idxs):
        # idxs (batch,seq_len)
        # 位置注意就是乘以根号d_model
        return self.emb(idxs) * math.sqrt(self.d_model)  # (batch,seq_len,d_model)
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,d_model,num_heads=8,num_encoder_layers=6,num_decoder_layers=6,dim_feedforward=2048,  dropout=0.1):
        super().__init__()
        self.src_input_layer = InputLayer(src_vocab_size, d_model)
        self.transform=nn.Transformer(d_model,nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,batch_first=True,dropout=dropout)
        self.tgt_input_layer = InputLayer(tgt_vocab_size, d_model)
        self.ln = nn.Linear(in_features=d_model,out_features=tgt_vocab_size) # 把d_model维度的向量映射到词表大小维度的“概率分布”向量

    def forward(self,src,tgt):
        src_input=self.src_input_layer(src)
        tgt_input=self.tgt_input_layer(tgt)
        mask = self.transform.generate_square_subsequent_mask(tgt_input.shape[1])
        transformer_output=self.transform(src_input,tgt_input,tgt_mask=mask)
        output=self.ln(transformer_output)
        return output

if __name__ == '__main__':
    src=torch.LongTensor([[1,2,3,4,5],[1,2,3,4,5]])
    tgt=torch.LongTensor([[1,2,3,4,5],[1,2,3,4,5]])
    seq2seq=Seq2SeqTransformer(src_vocab_size=10,tgt_vocab_size=12,d_model=128)
    seq = seq2seq(src, tgt)
    print( seq.shape)