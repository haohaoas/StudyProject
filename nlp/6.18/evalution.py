
import torch
from EnglishToFrenchOwnTokenizer import Tokenizer
import pandas as pd
'''
# 需求: 随机抽样5条数据，输入源语言句子，输出真实目标语言句子和预测目标语言句子 做个对比评估
# 步骤：
# 0. 导入必要的库
# 1. 加载分词器
# 2. 加载训练好的模型
# 3. 加载数据集，随机抽样5条数据
# 4. 遍历数据，提取源语言句子(打印用)、目标语言句子(打印用)和源语言索引序列(预测用)
# 5. 定义translate函数，接收源语言索引序列
#   5.1 将索引序列转换为tensor并添加batch维度
#   5.2 模型预测，获取目标语言预测结果(注意设置评估模式和禁用梯度)
#   5.3 对目标语言预测结果进行argmax操作，获取预测的索引序列
#   5.4 去掉<eos>和<pad>的索引
#   5.5 解码为目标语言字符串
# 6. 打印输入、真实目标语言句子和预测目标语言句子
'''

# 1. 加载分词器
en_tk = Tokenizer.load('model/en_tokenizer.pkl')
fr_tk = Tokenizer.load('model/fr_tokenizer.pkl')
# 2. 加载训练好的模型(注意把设备转移到cpu)
model = torch.load('model/model.pth',weights_only=False).to('cpu')

# 5. 定义translate函数，接收源语言索引序列
def translate(src_index):
    # 作用： 接收源语言索引去预测，最后用tokenizer.decode()反出目标语言字符串
    #   5.1 将索引序列转换为tensor并添加batch维度
    src_index_tensor = torch.LongTensor([src_index]) # (batch=1,src_len_len)
    model.eval()
    with torch.no_grad():
    #   5.2 模型预测，获取目标语言预测结果(注意设置评估模式和禁用梯度)
        tgt_pred = model(src_index_tensor)  #(batch=1,tgt_seq,tgt_vocab_size)
        tgt_pred = torch.squeeze(tgt_pred,0) # 挤压batch轴 (tgt_seq,tgt_vocab_size)
    #   5.3 对目标语言预测结果进行argmax操作，获取预测的索引序列
        tgt_pred_index = torch.argmax(tgt_pred,-1) #(tgt_seq)
        tgt_pred_index = tgt_pred_index.tolist() #变成列表
    #   5.4 去掉<eos>和<pad>的索引
        tgt_pred_index = [i for i in tgt_pred_index if i not in [0,2]]
    #   5.5 解码为目标语言字符串
        return fr_tk.decode(tgt_pred_index)


def evaluate():
    # 3. 加载数据集，随机抽样5条数据
    df = pd.read_csv('data/eng_fra.tsv',sep='\t')
    df = df.sample(5)
    # 4. 遍历数据，提取源语言句子(打印用)、目标语言句子(打印用)和源语言索引序列(预测用)
    for i in df.index:
        en = df['en'][i]
        fr = df['fr'][i]
        en_index = Tokenizer.str_to_list(df['en_index'][i])
        fr_pred = translate(en_index)
        print('en:',en)
        print('fr_true:',fr)
        print('fr_pred:',fr_pred)
        print('_'*50)

if __name__ == "__main__":
    # 执行评估
    evaluate()