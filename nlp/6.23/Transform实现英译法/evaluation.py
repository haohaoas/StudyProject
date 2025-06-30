from model import Seq2SeqTransformer
from tokenizer import Tokenizer
import pandas as pd

'''
- 作用：模型评估
- 自回归生成
  - 初始化为`<sos>`
  - 每次用当前序列预测下一个token, 把这个token与之前的序列拼起来，最初的序列就是一个<sos>
  - 循环生成直到`<eos>`或达到`max_len`,max_len最大序列长度只是为了万一模型始终预测的不到<eos>,能够停止。
- 输出处理
  - 跳过`<sos>`后解码为文本

改动细节：
1. 此次需先实例化模型，然后采取model.load_state_dict()加载模型参数
2. 此次没有索引化后的文件，所以加载源txt文件,注意设置列名,源序列索引用tokenizer的encode传入源序列
3. translate函数中实现自回归，注意设置max_len,防止万一预测不到<eos>索引
4. 注意transformer网络是并行输入，中间没有h传递信息，自回归时每次输入的是t时刻之间所有的token.
5. 初始化<sos>的索引，注意考虑batch轴 (batch=1,1)

6. 先用<sos>的索引预测，取预测序列的最后一个token索引拼接与<sos>拼接，如此反复每次得到的token都与之前拼接
7. 如果预测到<eos>或<pad>就停止
8. 输出注意去掉第一个<sos>
9. 全局注意形状，该挤压挤压，该扩轴扩轴
'''
import torch

en_tk = Tokenizer.load('data/en_tokenizer.pkl')
fr_tk = Tokenizer.load('data/fr_tokenizer.pkl')
params = torch.load('model/model.pth', map_location='mps')
model = Seq2SeqTransformer(src_vocab_size=len(en_tk), tgt_vocab_size=len(fr_tk), d_model=512)
model.load_state_dict(params)


df = pd.read_csv('data/test.csv', header=None)
df.columns = ['en', 'fr']
df = df.sample(5)


def translate(en_index, max_len=12):
    model.eval()
    src = torch.LongTensor([en_index])
    with torch.no_grad():
        tgt = torch.LongTensor([[1]])  # (batch=1,seq_len)
        for _ in range(max_len):
            tgt_pred = model(src, tgt)  # (batch=1,tgt_seq,tgt_vocab_size)
            tgt_pred_indexs = torch.argmax(tgt_pred, -1)  # (batch=1,tgt_seq=1)

            tgt_pred_index = tgt_pred_indexs[:, -1]  # (1) 取预测序列的最后一个token

            if tgt_pred_index.item() in [2, 0]:
                break
            tgt_pred_index = torch.unsqueeze(tgt_pred_index, 0)  # (batch=1,tgt_seq=1)

            tgt = torch.cat([tgt, tgt_pred_index], dim=1)
        tgt = tgt.squeeze(0)

        tgt_pred_index = tgt.tolist()

        return fr_tk.decode(tgt_pred_index[1:])


for i in df.index:
    en = df['en'][i]
    fr = df['fr'][i]
    # 2.1 用tokenizer的encode传入源序列 得到 源序列索引
    en_index = en_tk.encode(en)
    fr_pred = translate(en_index)
    print('en:', en)
    print('fr_true:', fr)
    print('fr_pred:', fr_pred)
    print('_' * 50)
