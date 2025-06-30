import torch
from torch.nn.utils.rnn import pad_sequence#用于处理文本长度不一致的问题
x= [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
           [2, 32, 1, 23, 1]]
# x=torch.tensor(x)#会报错，因为x不是两个列表的长度不一致
x = [torch.tensor(xi) for xi in x]
padded = pad_sequence(x, batch_first=True, padding_value=0, padding_side='left')#batch_first=True batch轴的位置在最前 padding_value 用0来填充不足的位置 padding_side='left'表示从左边开始填充
print(padded)


from sklearn.feature_extraction.text import CountVectorizer

text = ["我爱自然语言处理", "自然语言很有趣"]
vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='char')  # 提取 1-gram 和 2-gram 字符特征
X = vectorizer.fit_transform(text)

print(vectorizer.get_feature_names_out())