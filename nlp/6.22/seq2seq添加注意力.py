import math
import random

import torch
import torch.nn as nn

'''
需求: 实现一个无注意力的seq2seq模型
步骤:
1. 定义Encoder类
    1.1 定义网络层
        1.1.1 定义一个Embedding层
        1.1.2 定义一个GRU层
    1.2 定义前向传播函数
        1.2.1 根据输入序列的索引获取嵌入向量
        1.2.2 将嵌入向量传入GPU,获取GRU的输出序列张量输出

2. 定义Decoder类
    2.1 定义网络层
        2.1.1 定义一个Embedding嵌入层
        2.1.2 定义一个GRU单元
        2.1.3 定义一个线性层输出层
        2.1.4 注意把目标序列长度传入
    2.2 定义前向传播函数
        2.2.1 根据输入的Encoder序列张量求和,得到上下文向量C，注意轴度
        2.2.2 将上下文向量C当作初始隐藏状态h0
        2.2.3 定义一个批量大小的<SOS>的索引,作为初始x索引
        2.2.4 开始循环,循环次数为目标语言序列的最大长度
            2.2.4.1 将x索引传入嵌入层,得到嵌入向量
            2.2.4.2 将嵌入向量与h传入GRU单元,得到下一个h
            2.2.4.3 将h传入线性层,得到target_vocab_size的概率分布
            2.2.4.4 用一个output_list记录每个时间步的概率分布
            2.2.4.5 将概率分布传入argmax函数,得到预测的下一个词的索引, 注意轴度
            2.2.4.6 将预测的下一个词的索引当作x索引开始下一次循环
        2.2.5 将output_list中的每个output调整形状与数据类型，最后输出，注意轴度

3. 定义Seq2Seq类
    3.1 定义Encoder和Decoder
        3.1.1 定义Encoder
        3.1.2 定义Decoder
    3.2 定义前向传播函数
        3.2.1 将源语言序列传入Encoder,得到输出序列张量
        3.2.2 将Encoder的输出序列张量传入Decoder,得到目标语言序列的概率分布
        3.2.3 返回Decoder的输出
'''


# 1. 定义Encoder类
class Encoder(nn.Module):

    def __init__(self, src_vocab_size, emb_dim, h_dim):
        # src_vocab_size 源语言的词表大小
        super().__init__()
        # 1.1.1 定义一个Embedding层
        self.emb = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=emb_dim)
        # 1.1.2 定义一个GRU层
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=h_dim, batch_first=True)

    #    1.2 定义前向传播函数
    def forward(self, batch_src_tensor):
        # batch_src_tensor 代表源序列的索引张量 (batch,src_seq_len)
        #        1.2.1 根据输入序列的索引获取嵌入向量
        h = self.emb(batch_src_tensor)  # (batch,src_seq_len,e_dim)
        #        1.2.2 将嵌入向量传入GPU,获取GRU的输出序列张量输出
        outputs, _ = self.gru(h)  # outputs (batch, src_seq_len,h_dim)
        return outputs

class AttentionNetWork(nn.Module):
    def __init__(self,h_dim):
        super().__init__()
        self.WQ=nn.Linear(in_features=h_dim, out_features=h_dim)
        self.WK=nn.Linear(in_features=h_dim, out_features=h_dim)
        self.WV=nn.Linear(in_features=h_dim, out_features=h_dim)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, Q, encoder_outputs):
        Q=self.WQ(Q)
        K=self.WK(encoder_outputs)
        V=self.WV(encoder_outputs)
        Q=torch.unsqueeze(Q,dim=1)
        score=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(K.size(-1)) #计算qk的点积除以根号dk得到注意力分数
        attention_weights = self.softmax(score)
        outputs=torch.matmul(attention_weights,V)
        outputs=torch.squeeze(outputs,dim=1)
        return outputs


# 2. 定义Decoder类
class Decoder(nn.Module):

    #    2.1 定义网络层
    def __init__(self, tgt_vocab_size, emb_dim, h_dim, tgt_len):
        super().__init__()
        #        2.1.1 定义一个Embedding嵌入层
        self.emb = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=emb_dim)
        #        2.1.2 定义一个GRU单元
        self.gru_cell = nn.GRUCell(input_size=emb_dim, hidden_size=h_dim)
        #        2.1.3 定义一个线性层输出层
        self.l = nn.Linear(in_features=h_dim, out_features=tgt_vocab_size)
        #        2.1.4 注意把目标序列长度传入
        self.tgt_len = tgt_len
        self.attention_network=AttentionNetWork(h_dim)

    #    2.2 定义前向传播函数
    def forward(self, encoder_output, target_seq,teacher_forcing=0.5):
        # encoder_output (batch, src_seq_len, h_dim)

        h=encoder_output[:, -1, :]
        #        2.2.3 定义一个批量大小的<SOS>的索引,作为初始x索引,encoder_output.shape[0]获取batch_size
        x_ids = torch.full(size=(encoder_output.shape[0],), fill_value=1)  # (batch)
        output_list = []
        #        2.2.4 开始循环,循环次数为目标语言序列的最大长度
        for i in range(self.tgt_len):
            C = self.attention_network(Q=h, encoder_outputs=encoder_output)
            #           2.2.4.1 将x索引传入嵌入层,得到嵌入向量
            embed = self.emb(x_ids)  # (batch, emb_dim)
            #           2.2.4.2 将嵌入向量与h传入GRU单元,得到下一个h
            h = self.gru_cell(embed, C)  # (batch, h_dim)
            #           2.2.4.3 将h传入线性层,得到target_vocab_size的"概率分布"向量，又称逻辑向量
            logits = self.l(h)  # (batch,target_vocab_size)
            #           2.2.4.4 用一个output_list记录每个时间步的概率分布
            output_list.append(
                torch.unsqueeze(logits, dim=1))  # logits (batch,target_vocab_size) -> (batch, 1, target_vocab_size)
            #           2.2.4.5 将逻辑向量传入argmax函数,得到预测的下一个词的索引, 注意轴度
            if target_seq!=None and random.random()<teacher_forcing:
                pass
            else:
                x_ids = torch.argmax(logits, dim=-1)  # ( batch )
        #           2.2.4.6 将预测的下一个词的索引当作x索引开始下一次循环
        #       2.2.5 将output_list中的每个output调整形状与数据类型，最后输出，注意轴度
        outputs = torch.cat(output_list, dim=1)  # (batch,tgt_len,target_vocab_size)
        return outputs


# 3. 定义Seq2Seq类
class Seq2Seq(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim, h_dim, tgt_len):
        super().__init__()
        # 3.1.1 定义Encoder
        self.encoder = Encoder(src_vocab_size, emb_dim, h_dim)
        # 3.1.2 定义Decoder
        self.decoder = Decoder(tgt_vocab_size, emb_dim, h_dim, tgt_len)

    #    3.2 定义前向传播函数
    def forward(self, batch_src_tensor,target_seq=None,teacher_forcing=0.5):
        # batch_src_tensor (batch,src_len)
        #    3.2.1 将源语言序列传入Encoder,得到输出序列张量
        encoder_output = self.encoder(batch_src_tensor)  # (batch,src_len,h_dim)
        #        3.2.2 将Encoder的输出序列张量传入Decoder,得到目标语言序列的概率分布
        outputs = self.decoder(encoder_output,target_seq,teacher_forcing)  # (batch,tgt_len,tgt_vocab_size)
        #        3.2.3 返回Decoder的输出
        return outputs


if __name__ == '__main__':
    source_vocabs_num = 100  # 源语言词汇表大小
    target_vocabs_num = 110  # 目标语言词汇表大小
    hidden_dim = 32  # 隐藏层维度
    emb_dim = 16  # 词嵌入维度
    batch_size = 8  # 批大小
    source_max_len = 10  # 源语言序列的最大长度
    target_max_len = 12  # 目标语言序列的最大长度

    encoder = Encoder(source_vocabs_num, emb_dim, hidden_dim)

    model = Seq2Seq(source_vocabs_num, target_vocabs_num, emb_dim, hidden_dim, target_max_len)
    source_seqence_indexes = torch.randint(0, source_vocabs_num,
                                           (batch_size, source_max_len))  # [batch_size, source_max_len]
    outputs = model(source_seqence_indexes, None,0.6)  # [batch, target_max_len, vocab_size]
    print("正确的形状应该为:", [batch_size, target_max_len, target_vocabs_num])
    print("[batch, target_max_len, target_vocabs_num]:", outputs.shape)  # [batch, target_max_len, target_vocabs_num]