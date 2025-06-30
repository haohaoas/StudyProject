import torch
import torch.nn as nn
import torch.optim as optim
#RNN预测文本，测试文本遗忘
# 构造数据
# def generate_batch(batch_size=32, seq_len=40):
#     xs = []
#     ys = []
#     for _ in range(batch_size):
#         x = torch.zeros(seq_len)
#         pos = torch.randint(low=0, high=seq_len, size=(1,)).item()
#         x[pos] = 1.0
#         xs.append(x)
#         ys.append(pos)
#     return torch.stack(xs), torch.tensor(ys)
#
# class SimpleRNN(nn.Module):
#     def __init__(self,  input_size=1, hidden_size=8,batch_fisrt=True):
#         super(SimpleRNN, self).__init__()
#         self.rnn = nn.GRU(input_size, hidden_size, batch_first=batch_fisrt)
#         self.fc = nn.Linear(hidden_size, 1)
#     def forward(self, x):
#         x=x.unsqueeze(-1)
#         out,_ = self.rnn(x)
#         out = self.fc(out[:,-1,:])
#         return out.squeeze(-1)
# model=SimpleRNN()
# criterion=nn.MSELoss()
# optimizer=optim.Adam(model.parameters(),lr=0.01)
# for epoch in range(100):
#     model.train()
#     x, y = generate_batch(seq_len=20)
#     pred=model(x)
#     loss=criterion(pred,y.float())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}: loss={loss.item():.4f}, pred[0]={pred[0].item():.2f}, true[0]={y[0].item()}")

#LSTM前向传播模拟
input_size=4#输入维度
hidden_size=3#隐藏层大小
def init_weights():
    return {
        'W_f':nn.Parameter(torch.randn(hidden_size,input_size)),
        'U_f':nn.Parameter(torch.randn(hidden_size,hidden_size)),
        'b_f':nn.Parameter(torch.randn(hidden_size)),

        'W_i':nn.Parameter(torch.randn(hidden_size,input_size)),
        'U_i':nn.Parameter(torch.randn(hidden_size,hidden_size)),
        'b_i':nn.Parameter(torch.randn(hidden_size)),

        'W_c':nn.Parameter(torch.randn(hidden_size,input_size)),
        'U_c':nn.Parameter(torch.randn(hidden_size,hidden_size)),
        'b_c':nn.Parameter(torch.randn(hidden_size)),

        'W_o':nn.Parameter(torch.randn(hidden_size,input_size)),
        'U_o':nn.Parameter(torch.randn(hidden_size,hidden_size)),
        'b_o':nn.Parameter(torch.randn(hidden_size)),
    }

params=init_weights()