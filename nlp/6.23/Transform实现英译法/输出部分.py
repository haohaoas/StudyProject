import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, d_model, tgt_vocab_size):
        super().__init__()
        #d_model 模型约束的传输向量维度 tgt_vocab_size 目标词表大小
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

if __name__ == '__main__':
    x=torch.randn(3,6,512)
    model=Generator(512,10000)
    print(model(x).shape)
    print(model(x))