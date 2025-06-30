import time

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 第一步：加载 bert 分词器（AutoTokenizer）
# 指定要使用的模型名称，例如 "bert-base-uncased"
# 这个分词器会将文本转为 input_ids、attention_mask 等
device=torch.device( 'cpu')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 第二步：加载 train.csv 数据（Pandas）
df = pd.read_csv('train_data.csv')
# 提取 text 和 label 两列
text=df['text']
label=df['label']
text=list(text)
# 使用 tokenizer 批量编码文本（tokenizer.batch_encode_plus）
encoded= tokenizer(text, truncation=True, padding=True, return_tensors="pt")
# 设置 truncation=True，padding=True，return_tensors="pt"
# 返回值是一个字典，包含 input_ids、attention_mask 等

# 第三步：构建 PyTorch Dataset 对象
labels = torch.tensor(label.values, dtype=torch.long)
dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'], labels)
# 将 input_ids、attention_mask、labels 封装为 TensorDataset
# 每个都是 torch.tensor，并注意 dtype=torch.long
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# 然后使用 DataLoader 批量加载数据，设置 batch_size 和 shuffle=True

# 第四步：加载预训练 BERT 模型（AutoModelForSequenceClassification）
sequence_classification = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# 设置 num_labels=2，因为是二分类任务
# 这个模型已经包含了分类头，直接用于 fine-tune
sequence_classification.to( device)
# 第五步：定义优化器（AdamW）和损失函数（CrossEntropyLoss）
optimizer = optim.AdamW(sequence_classification.parameters(), lr=0.0001)
loss = nn.CrossEntropyLoss()
# 设置训练轮数（如 epochs=3）
epochs=4
sequence_classification.train()
# 启动训练循环：
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for epoch in range(epochs):
    total_loss = 0
    start_time = time.time()  # 记录时间
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        with autocast():
            output = sequence_classification(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Time: {time.time() - start_time:.2f}s", flush=True)
# 每轮结束后打印 loss 作为监控指标


# 第六步：在验证集上评估模型性能
valid_df = pd.read_csv('valid_data.csv')
texts=list(valid_df['text'])
labels=torch.tensor(valid_df['label'].values,dtype=torch.long)
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
dataset=TensorDataset(encoded['input_ids'], encoded['attention_mask'], labels)
# 加载 valid.csv 并进行相同的分词处理
data_loader = DataLoader(dataset, batch_size=128)
# 构建 DataLoader（不需要 shuffle）
# 4. 在验证集上评估模型
correct=0
total=0
sequence_classification.eval()
with torch.no_grad():
    for batch in data_loader:
        inputs, masks, labels = batch
        outputs = sequence_classification(inputs, masks)
        predictitions = torch.argmax(outputs.logits, dim=1)
        correct += (predictitions == labels).sum().item()
        total += labels.size(0)
# 在不更新梯度的条件下进行预测（with torch.no_grad()）
accuracy = correct / total
# 比较预测结果和真实标签，计算准确率（accuracy）
print(f"Validation Accuracy: {accuracy:.4f}")