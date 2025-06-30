import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 构造词表和示例数据
word2idx = {'小明': 0, '吃': 1, '苹果': 2, '没': 3}
sentences = [
    ['小明', '没', '吃', '苹果'],  # 否定，label 0
    ['小明', '吃', '苹果'],      # 肯定，label 1
]
labels = [0, 1]

# 模型定义
class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.W_Q = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        Q = self.W_Q(embedded)  # [batch_size, seq_len, hidden_dim]
        K = self.W_K(embedded)  # [batch_size, seq_len, hidden_dim]
        V = self.W_V(embedded)  # [batch_size, seq_len, hidden_dim]

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(K.size(-1))  # [batch_size, seq_len, seq_len]
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)  # [batch_size, seq_len, hidden_dim]

        pooled = torch.mean(attended, dim=1)  # [batch_size, hidden_dim]
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        return logits

# 示例测试
model = AttentionClassifier(vocab_size=4, embed_dim=3, hidden_dim=2, num_classes=2)
# 测试单条数据时需加 batch 维
sentence = torch.tensor([word2idx[word] for word in sentences[0]])
logits = model(sentence.unsqueeze(0))
print("logits:", logits)

def encode_sentence(sentence,word2idx,max_len):
    return [word2idx.get(word,0) for word in sentence] + [0] * (max_len - len(sentence))
max_len=4
epochs=100
batch_size=2
X = torch.stack([torch.tensor(encode_sentence(sentence, word2idx, max_len)) for sentence in sentences])
y=torch.tensor(labels)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
dataset = TensorDataset(X, y)
dataloader=DataLoader(dataset,batch_size=2)
for epoch in range(epochs):
    total_loss=0
    for X, y in dataloader:
        optimizer.zero_grad()
        logits = model(X)    /
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch:", epoch + 1, "Loss:", total_loss / len(dataloader))
