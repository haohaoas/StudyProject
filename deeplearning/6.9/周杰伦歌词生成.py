import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import  re
import jieba
from torch.utils.data import DataLoader
def perprocess():
    fileName='jaychou_lyrics.txt'
    unique_words = []
    all_words = []
    # 遍历数据集中的每一行文本
    for line in open(fileName, 'r'):
        # 使用jieba分词,分割结果是一个列表
        words = jieba.lcut(line)
        # print(words)
        # 所有的分词结果存储到all_sentences，其中包含重复的词组
        all_words.append(words)
    # 遍历分词结果，去重后存储到unique_words
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    # 语料中词的数量
    word_count = len(unique_words)
    word_to_index = {word: idx for idx, word in enumerate(unique_words)}
    # 词表索引表示
    corpus_idx = []
    # 遍历每一行的分词结果
    for words in all_words:
        temp = []
    # 获取每一行的词，并获取相应的索引
        for word in words:
            temp.append(word_to_index[word])
    # 在每行词之间添加空格隔开
            temp.append(word_to_index[' '])
    # 获取当前文档中每个词对应的索引
        corpus_idx.extend(temp)
    return unique_words, word_to_index, word_count, corpus_idx
if __name__ == '__main__':
    word_count,corpus_idx,unique_words,word_to_idx=perprocess()
    print("词的数量：\n",word_count)
    print("去重后的词:\n",unique_words)
    print("每个词的索引：\n",word_to_idx)
    print("当前文档中每个词对应的索引：\n",corpus_idx)

class lyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        self.word_count=len(self.corpus_idx)
        self.number=self.word_count//self.num_chars
    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start=min(max(idx,0),self.word_count-self.num_chars-2)
        #确保 start 的值在合理范围内，
        # 不小于 0，不大于 self.word_count - self.num_chars - 2，
        # 以避免索引越界或无效
        x=self.corpus_idx[start:start+self.num_chars]
        y=self.corpus_idx[start+1:start+self.num_chars+1]
        return torch.tensor(x),torch.tensor(y)
if __name__ == '__main__':
    dataset = lyricsDataset(corpus_idx, num_chars=5)
    x,y=dataset.__getitem__(0)
    print(f'网络输入值{x}')
    print(f'网络输出值{y}')