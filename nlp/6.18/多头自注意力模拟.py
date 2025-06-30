import math

import torch
import torch.nn as nn

batch_size = 2#批次
seq_len = 4#序列长度
embed_dim = 8 #词向量维度
num_heads = 2#多头注意力的头数
head_dim = embed_dim // num_heads  # 4

x = torch.rand(batch_size, seq_len, embed_dim)  # 模拟词向量输入

W_q = nn.Linear(embed_dim, embed_dim)
W_k = nn.Linear(embed_dim, embed_dim)
W_v = nn.Linear(embed_dim, embed_dim)

Q = W_q(x)
K = W_k(x)
V = W_v(x)


def split_heads(x, num_heads):#对注意力进行分头
    batch_size, seq_len, embed_dim = x.size() # 获取输入的形状
    head_dim = embed_dim // num_heads # 计算每个头的维度
    # 变成 [batch_size, seq_len, num_heads, head_dim]
    x = x.view(batch_size, seq_len, num_heads, head_dim)
    # 转置成 [batch_size, num_heads, seq_len, head_dim]
    return x.transpose(1, 2)


Q = split_heads(Q, num_heads)#切割后的 Q
K = split_heads(K, num_heads)#切割后的 K
V = split_heads(V, num_heads)#切割后的 V

dk=K.size(-1)#获取k的最后一个维度的大小
attention_score =torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(dk) #计算qk的点积除以根号dk得到注意力分数
weights = torch.softmax(attention_score, dim=-1)#计算权重
outputs=torch.matmul(weights,V)#得到经过注意力机制后的新表示
def combine_heads(x):#合并多头
    batch_size, num_heads, seq_len, head_dim = x.size()#获取输入的形状
    # 转置成 [batch_size, seq_len, num_heads, head_dim]
    x = x.transpose(1, 2)
    # 改变形状为 [batch_size, seq_len, embed_dim]
    return x.contiguous().view(batch_size, seq_len, num_heads * head_dim)
outputs = combine_heads(outputs)
outputs_linear = nn.Linear(embed_dim, embed_dim)#线性映射
outputs = outputs_linear(outputs)
print( outputs)