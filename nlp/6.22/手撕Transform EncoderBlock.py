import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerEncoderBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,ffn_hidden_dim,dropout=0.1):#词向量维度，头数，FFN隐藏层维度，dropout概率
        super(TransformerEncoderBlock, self).__init__()
        #多头注意力机制
        self.mha=nn.MultiheadAttention(embedding_dim,num_heads,batch_first= True)
        #FFN
        self.fnn=nn.Sequential(
            nn.Linear(embedding_dim,ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim,embedding_dim)
        )
        #残差连接
        self.ln1=nn.LayerNorm(embedding_dim)
        self.ln2=nn.LayerNorm(embedding_dim)

        #dropout
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x):
        #多头注意力机制
        attn_output,_ = self.mha(x, x, x)
        x=self.dropout1(attn_output)+ x
        x=self.ln1(x)
        #FFN
        ffn_output=self.fnn(x)
        x=self.dropout2(ffn_output)+ x
        x=self.ln2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,embedding_dim,num_heads,ffn_hidden_dim,num_layers,dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.blocks=nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim,num_heads,ffn_hidden_dim,dropout
            ) for _ in range(num_layers)
        ])

    def forward(self,x):
        for block in self.blocks:
            x=block(x)
        return x



