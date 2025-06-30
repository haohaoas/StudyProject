import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.xh2gates= nn.Linear(input_size+hidden_size, 4 * hidden_size)#4个隐藏层
    def forward(self, x,hx,cx):#当前时间步的输入向量，隐藏状态hx，记忆状态cx
        combs = torch.cat([x, hx], dim=1)#拼接输入和上一个隐藏状态
        gates= self.xh2gates(combs)
        i,f,g,o = gates.chunk(4, dim=1) # 将gates拆分成4个部分
        i = torch.sigmoid(i)#输入门
        f = torch.sigmoid(f)#遗忘门
        g = torch.tanh(g)#更新门
        o = torch.sigmoid(o)#输出门
        c_next= f * cx + i * g#f*cx f是0-1的值，乘记忆单元表示保留旧记忆的程度 i是控制是否允许信息加入记忆 g是得到的新候选信息
        h_next= o * torch.tanh(c_next)#当前输出=输出门*tanh(更新门)
        return c_next, h_next

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):#输入维度，隐藏层维度，输出维度
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        hx = torch.zeros(inputs.size(0), self.hidden_size)
        cx = torch.zeros(inputs.size(0), self.hidden_size)
        inputs = inputs.permute(1, 0, 2)
        outputs = []
        for x in inputs:
            cx, hx = self.cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))
        out = self.linear(hx)
        return out

if __name__ == '__main__':
    seq_len=5
    input_size=10
    hidden_size=6
    num_classes=3
    batch_size=4
    num_epochs=20
    model = LSTM(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X=torch.randn(100,seq_len,input_size)#样本数，序列长度，输入维度
    y=torch.randint(0,num_classes,(100,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        total_loss=0
        for batch_x,batch_y in data_loader:
            outpus=model(batch_x)
            loss=criterion(outpus,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch:", epoch + 1, "Loss:", total_loss / len(data_loader))