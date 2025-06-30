import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GruCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.zt = nn.Linear(input_size + hidden_size, hidden_size)
        self.rt = nn.Linear(input_size + hidden_size, hidden_size)
        self.ht = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=-1)  # 拼接当前输入x和重置门过的上一隐藏状态
        zt = torch.sigmoid(self.zt(combined))  # 更新门
        rt = torch.sigmoid(self.rt(combined))  # 重置门
        combine_r = torch.cat([x, rt * h_prev], dim=-1)  # 拼接重置门后的上一隐藏状态和当前输入x
        ht = torch.tanh(self.ht(combine_r))  # 候选隐藏状态
        h_next = (1 - zt) * h_prev + zt * ht  # 更新后的隐藏状态
        return h_next
class Gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = GruCell(input_size, hidden_size)
    def forward(self,xs):
        batch_size,seq_len,_ = xs.size()
        h=torch.zeros(batch_size,self.cell.hidden_size)
        outputs=[]
        for x in range(seq_len):
            x_t=xs[:,x,:]
            h=self.cell(x_t,h)
            outputs.append(h.unsqueeze(1))
        t = torch.cat(outputs, dim=1)
        return t, h

if __name__ == '__main__':
    batch_size = 10
    seq_len=20
    input_size = 100
    hidden_size = 200
    gru = Gru(input_size, hidden_size)
    x = torch.randn(batch_size, seq_len, input_size)
    y=torch.randn(batch_size,seq_len,hidden_size)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
    entropy_loss = nn.MSELoss()
    epochs = 100
    for i in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output, h = gru(batch_x)
            loss = entropy_loss(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {i + 1}/{epochs}, Loss: {total_loss}")