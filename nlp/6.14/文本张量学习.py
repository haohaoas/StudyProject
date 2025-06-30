import torch
import torch.nn as nn
from networkx.classes import neighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 创建一个简单的 Embedding 层（10个词，每个5维向量）
# embedding = nn.Embedding(num_embeddings=10, embedding_dim=5)
#
# # 获取词索引 0~9 的嵌入向量
# word_indices = torch.arange(0, 10)
# embeddings = embedding(word_indices).detach().numpy()
#
# # PCA降维到2D
# pca = PCA(n_components=2)
# reduced = pca.fit_transform(embeddings)
#
# # 可视化向量
# plt.figure(figsize=(8, 6))
# for i, (x, y) in enumerate(reduced):
#     plt.scatter(x, y)
#     plt.text(x + 0.01, y + 0.01, f"word_{i}", fontsize=10)
# plt.title("Embedding 向量的 PCA 可视化（随机初始化）")
# plt.grid(True)
# plt.show()
import fasttext
#
# model = fasttext.train_unsupervised('fil9.txt')
# model.save_model('fasttext_model.bin')
model = fasttext.load_model('fasttext_model.bin')
vector = model.get_word_vector('cat')
print(vector)
neighbor = model.get_nearest_neighbors('cat', k=5)
print(neighbor)
