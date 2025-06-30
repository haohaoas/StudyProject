from typing import Counter

import pandas as pd
'''
作用：切分训练集测试集
先随机打乱，之后切片即可。
具体步骤：
1. 打乱：df = df.sample(frac=1)实现全量打乱
2. 得到一个数字作为切片位置，该数字代表切片位置。可通过总数据量*0.9获取该数字
3. train = df[:切片位置]， test = df[切片位置:]即可
4. 最后分别保存训练集和测试集
'''
df = pd.read_csv('data/eng-fra-v2.txt', sep='\t', header=None)
df = df.sample(frac=1)
train = df[:int(len(df) * 0.8)]
test=df[int(len(df) * 0.8):]
train.to_csv('data/train.txt', header=None, index=False, sep='\t')
test.to_csv('data/test.txt', header=None, index=False, sep='\t')
