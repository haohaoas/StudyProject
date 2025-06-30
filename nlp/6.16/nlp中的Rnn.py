import torch
import torch.nn as nn
#
# # 假设我们要处理一个 batch 中 2 句话，每句5个词，每个词是300维向量
# batch_size = 2
# seq_len = 5
# input_dim = 300
# hidden_size = 128
#
# # 构造输入数据 [batch_size, seq_len, input_dim]
# x = torch.randn(batch_size, seq_len, input_dim)
#
# # 构造标签（假设是每个词都要分类成10类）
# y = torch.randint(0, 10, (batch_size, seq_len))  # [2, 5]
#
# # 定义模型
# rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
# fc = nn.Linear(hidden_size, 10)  # 输出层：映射到10个分类
#
# # 定义损失函数
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(list(rnn.parameters()) + list(fc.parameters()), lr=0.00001)
# for epoch in range(5000):
#     optimizer.zero_grad()
#     output, _ = rnn(x)
#     logits = fc(output)  # 使用所有时间步的输出
#     loss = criterion(logits.view(-1, 10), y.view(-1))  # 调整形状匹配
#     print("loss:", loss.item())
#     loss.backward()
#     optimizer.step()
#     pred = torch.argmax(logits, dim=-1)  # 取出每个时间步的预测类别
#     print("模型预测结果 (pred):")
#     print(pred)
#
#     print("\n真实标签 (true y):")
#     print(y)
seq_len =6
dim=5
randn = torch.randn(seq_len, dim)
rnn = nn.RNN(input_size=dim, hidden_size=3, batch_first=True)
output,_ = rnn(randn)
print("output shape is ", output.shape)
print("output is ", output)
print("output[0] is ", output[0])