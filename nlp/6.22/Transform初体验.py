import hanlp
import torch
import torch.nn as nn

vocab_size=1000
embedding_dim=512
embedding=nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
x = torch.tensor([[1, 5, 23, 67, 2, 9],
                  [45, 87, 4, 999, 31, 8]])
token_embedding= embedding(x)
# 位置编码
# class PositionEncoding(nn.Module):
#     def __init__(self, embedding_dim=512, max_len=5000, d_model=512):
#         super().__init__()
#         pe = torch.zeros(max_len, embedding_dim)  # 初始化位置编码张量，形状 [max_len, embedding_dim]
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 形状 [max_len, 1]
#         div_term = 10000.0 ** (torch.arange(0, d_model, 2) / d_model)
#         pe[:, 0::2] = torch.sin(position / div_term)
#         pe[:, 1::2] = torch.cos(position / div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)#把pe加入模型中，不会被当作参数更新
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]#[batch_size, seq_len, embedding_dim]x+自己的位置向量
#         return x
#
# pos_encoder = PositionEncoding()
# encoded = pos_encoder(token_embedding)

#多头自注意力
def split_heads(query, num_heads):
    query = query.view(query.shape[0], query.shape[1], num_heads, -1)

# 多头自注意力
# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         assert embed_dim % num_heads == 0 # 确保嵌入维度可以被注意力头数整除
#
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads # 每个注意力头的维度
#         self.W_q = nn.Linear(embed_dim, embed_dim)
#         self.W_k = nn.Linear(embed_dim, embed_dim)
#         self.W_v = nn.Linear(embed_dim, embed_dim)
#         self.fc = nn.Linear(embed_dim, embed_dim)
#     def forward(self, x):
#         query = self.W_q(x)
#         key = self.W_k(x)
#         value = self.W_v(x)
#         query = split_heads(query, self.num_heads)
#         key = split_heads(key, self.num_heads)
#         value = split_heads(value, self.num_heads)
#         query = query.transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]
#         key = key.transpose(1, 2)# [batch_size, num_heads, seq_len, head_dim]
#         value = value.transpose(1, 2)# [batch_size, num_heads, seq_len, head_dim]
#         scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)



