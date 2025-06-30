import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, num_heads, ffn_hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding=nn.Embedding(src_vocab_size,embedding_dim)
        self.positional_encoding=PositionEncoding(embedding_dim)
        self.encoder=TransformerEncoder(embedding_dim,num_heads,ffn_hidden_dim,num_layers,dropout)
        self.decoder=TransformerDecoder(embedding_dim,num_heads,ffn_hidden_dim,num_layers,dropout)
        self.linear=nn.Linear(embedding_dim,src_vocab_size)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        enc_output = self.encoder(src, src_key_padding_mask)
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        dec_output = self.decoder(tgt, enc_output, src_key_padding_mask, tgt_mask)
        output = self.linear(dec_output)
        output = self.softmax(output)
        return output

class PositionEncoding(nn.Module):
    def __init__(self, embedding_dim=512, max_len=5000, d_model=512):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)  # 初始化位置编码张量，形状 [max_len, embedding_dim]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 形状 [max_len, 1]
        div_term = 10000.0 ** (torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)#选取所有行、从第0列开始每隔一列的列（即偶数列
        pe[:, 1::2] = torch.cos(position / div_term)#选取所有行、从第1列开始每隔一列的列
        pe = pe.unsqueeze(0) #形状 [1, max_len, embedding_dim]
        self.register_buffer('pe', pe)#把pe加入模型中，不会被当作参数更新

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]#[batch_size, seq_len, embedding_dim]x+自己的位置向量
        return x

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


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn=nn.MultiheadAttention(embedding_dim,num_heads,batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embedding_dim)
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, src_key_padding_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention
        self_attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # 2. Cross-Attention
        cross_attn_output, _ = self.cross_attn(x, encoder_outputs, encoder_outputs,
                                               key_padding_mask=src_key_padding_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        # 3. FeedForward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dim, num_heads, ffn_hidden_dim, dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_outputs, src_key_padding_mask=None, tgt_mask=None):
        for block in self.blocks:
            x = block(x, encoder_outputs, src_key_padding_mask, tgt_mask)
        return x


if __name__ == '__main__':
    src=torch.randn(10, 32, 512)
    tgt=torch.randn(10, 32, 512)
    transformer = Transformer(32, 512, 8, 2048, 6, 0.1)
    transformer1 = transformer(src, tgt)
    print(transformer1.shape)

