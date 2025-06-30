# 0. 导包torch, pandas, pad_sequence, Tokenizer
from EnglishToFrenchOwnTokenizer import Tokenizer
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
# 1. 加载英语和法语的分词器
en_tk=Tokenizer.load('model/en_tokenizer.pkl')
fr_tk=Tokenizer.load('model/fr_tokenizer.pkl')
# 2. 读取文件，添加列名
df = pd.read_csv('data/eng-fra-v2.txt', sep='\t', encoding='utf-8')
df.columns=['English', 'French']
# 3. 对英语和法语句子进行索引化处理，可使用dataframe的apply方法，注意法语句子需要添加结束符号2
df['English_index']=df['English'].apply(lambda x: en_tk.encode(x))
df['French_index']=df['French'].apply(lambda x: fr_tk.encode(x)+[2])

# 将 None 替换为 0（未知/填充标记）
df['English_index'] = df['English_index'].apply(lambda seq: [tok if tok is not None else 0 for tok in seq])
df['French_index'] = df['French_index'].apply(lambda seq: [tok if tok is not None else 0 for tok in seq])

print(df.head())
# 4. 将索引化后的句子转换为tensor
en_tensor_list = [torch.tensor(x) for x in df['English_index'].tolist()]
fr_tensor_list = [torch.tensor(x) for x in df['French_index'].tolist()]
# 5. 使用 pad_sequence 填充，并转换为列表形式便于存入 DataFrame
padded_en = pad_sequence(en_tensor_list, batch_first=True, padding_value=0, padding_side='left')
padded_fr = pad_sequence(fr_tensor_list, batch_first=True, padding_value=0)

# 将每行 tensor 转成 list，避免 DataFrame 形状冲突
df['English_index'] = padded_en.tolist()
df['French_index']  = padded_fr.tolist()
# 6. 将处理后的数据保存为tsv文件
df.to_csv('data/eng-fra.csv', sep='\t', index=False)

print(df.head())
