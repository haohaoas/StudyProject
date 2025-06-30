import time

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import ownTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据预处理函数
def preprocess_name_classification(file_path: str, token_level: str = "char", padding_value: int = 0, max_len: int = None):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["Name", "Country"])

    # 编码 Name
    name_tokenizer = ownTokenizer.Tokenizer(token_level=token_level, lang="en")
    name_tokenizer.build_vocab(df["Name"].tolist())
    name_id_list = name_tokenizer.encode(df["Name"].tolist())
    name_tensor_list = [torch.tensor(seq, dtype=torch.long) for seq in name_id_list]
    if max_len:
        name_tensor_list = [t[:max_len] for t in name_tensor_list]
    name_padded = pad_sequence(name_tensor_list, batch_first=True, padding_value=padding_value)
    df["name_index"] = name_padded.tolist()

    # 编码 Country
    label_tokenizer = ownTokenizer.Tokenizer(token_level="word", lang="en")
    label_tokenizer.build_vocab(df["Country"].tolist())
    country_id_list = label_tokenizer.encode(df["Country"].tolist())
    df["country_index"] = [int(t[0]) for t in country_id_list]

    vocab_size = max([token for seq in name_id_list for token in seq]) + 1
    output_size = max(df["country_index"]) + 1

    return df, name_tokenizer, label_tokenizer, vocab_size, output_size

# Dataset 类
class NameDataset(Dataset):
    def __init__(self, df):
        self.names = [torch.tensor(x) for x in df["name_index"]]
        self.labels = torch.tensor(df["country_index"].values)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx], self.labels[idx]

# 模型定义
class CharRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.rnn(x)
        out = self.fc(hn.squeeze(0))
        return out

# 主程序
df, name_tokenizer, label_tokenizer, vocab_size, output_size = preprocess_name_classification(
    "name_classfication.txt",
    token_level="char",
    padding_value=0,
    max_len=20
)

model = CharRNNClassifier(
    vocab_size=vocab_size,
    embedding_dim=64,
    hidden_size=128,
    output_size=output_size
).to(device)
def evaluate(model, data_loader):
    model.eval() # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad(): # 禁用梯度计算
        #1 遍历数据
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            #inputs: [batch_size, seq_len]
            #labels: [batch_size]
            #2 前向传播
            outputs = model(inputs) #[batch_size, label_num]
            #3 取最后输出向量中最大值所在的索引作为预测结果
            _, predicted = torch.max(outputs.data, 1) # [batch_size]
            #4 与标签进行对比得到正确数
            correct += (predicted == labels).sum().item()
            #5 累加正确数和总数,计算准确率
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    epochs=100
    batch_size=256
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = NameDataset(df)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        start_time=time.time()
        model.train()
        for names, labels in data_loader:
            names = names.to(device)
            labels = labels.to(device)
            preds = model(names)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time=time.time()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f},Spend time:{end_time-start_time :.4f}")
        train_accuracy=evaluate(model,data_loader)
        print(f"Train Accuracy: {train_accuracy:.4f}")

    # 保存模型
    torch.save(model, "char_rnn_classifier.pth")
#
# df, name_tokenizer, label_tokenizer = preprocess_name_classification(
#     "name_classfication.txt",
#     token_level="char",
#     padding_value=0,
#     max_len=20
# )
#
# # 保存为新 CSV
# df.to_csv("name_classification_processed.csv", sep="\t", index=False)
