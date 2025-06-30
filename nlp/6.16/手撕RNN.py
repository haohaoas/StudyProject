import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.W=nn.Linear(input_size+hidden_size,hidden_size)

    def forward(self, x, h):
        cat = torch.cat([x, h], dim=-1)
        h_t=torch.tanh(self.W(cat))
        return h_t
class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = RNNCell(input_size, hidden_size, output_size)

    def forward(self, xs):
        h = torch.zeros(batch_size, self.hidden_size)
        y = None
        for t in range(seq_len):
            x_t = xs[:, t, :]  # [batch_size, input_size]
            y, h = self.cell(x_t, h)  # y: [batch_size, output_size], h: [batch_size, hidden_size]
        return y, h  # y: [batch_size, output_size] (last time step)


seq_len = 5
input_size = 10
hidden_size = 20
output_size = 3
num_epochs = 100
batch_size = 10
criterion = nn.CrossEntropyLoss()
train_data = torch.randn(1000, seq_len, input_size)# [1000, 5, 10]
train_labels = torch.randint(0, output_size, (1000,))# [1000]
dataset =TensorDataset(train_data, train_labels)
data_loader = DataLoader(dataset, batch_size=batch_size)
rnn = RNN(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(list(rnn.parameters())+list(rnn.cell.parameters()), lr=0.001)

for epoch in range(num_epochs):
    for x_seq, label in data_loader:
        y, h = rnn(x_seq)
        print("y", y)
        print("h", h)
        print(y.shape)
        loss = criterion(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {}, Loss: {}".format(epoch, loss.item()))