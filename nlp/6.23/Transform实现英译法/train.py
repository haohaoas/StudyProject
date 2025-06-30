import torch
from torch import nn, optim
from tqdm import tqdm
from tokenizer import Tokenizer
from mydataset import  get_dataloader

from model import Seq2SeqTransformer
'''
作用：模型训练主脚本
3个声明: 模型,损失函数,优化器
2个加载: 加载tokenizer,加载数据集
2个循环: epoch循环, 数据batch遍历循环
- 遍历数据时进行5个步骤
  1 前向传播 
  2 计算损失 
  3 梯度清零
  4 反向传播 
  5 更新模型参数
关键细节：
- 解码器输入：目标序列去掉最后一个token (tgt[:, :-1]) , 去掉<eos>
- 解码器目标：目标序列去掉第一个token (tgt[:, 1:]), 去掉<sos>
其他补充：
1. 注意dataloader这次要传tokenizer
2. 注意文件路径及一些变量变一变
3. 额外新技巧，此次只保存模型参数. torch.save(model.state_dict(), 路径)
    优势：1. 体积更小。  2. 保护知识产权(只有参数，没有网络层定义)
'''
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
en_tk = Tokenizer.load('data/en_tokenizer.pkl')
fr_tk = Tokenizer.load('data/fr_tokenizer.pkl')
model = Seq2SeqTransformer(
    src_vocab_size=len(en_tk),
    tgt_vocab_size=len(fr_tk),
    d_model=512
).to( device)
criterion=nn.CrossEntropyLoss(ignore_index=0)
optimizer=optim.AdamW(model.parameters(),lr=0.0001)
train_loader=get_dataloader('data/train.csv',en_tk,fr_tk,batch_size=256,shuffle=True)
epochs=20
for epoch in range(epochs):
    model.train()
    total_loss=0
    for src,tgt in tqdm(train_loader):
        src,tgt=src.to(device),tgt.to(device)
        logits=model(src,tgt[:,:-1])
        logits_reshaped=logits.reshape(-1,logits.shape[-1])
        tgt_flat=tgt[:,1:].reshape(-1)
        loss_value=criterion(logits_reshaped,tgt_flat)
        total_loss += loss_value.item()
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    print(f'epoch:{epoch},平均loss:{total_loss / len(train_loader):.4f}')  # :.4f指的保留4位小数
torch.save(model.state_dict(), f'model/model.pth')