import ownTokenizer as tokenizer
import torch
from 人名分类器 import CharRNNClassifier
import torch.serialization

'''
需求：加载模型进行预测人名预测，预测出最有可能的3个类别

步骤：
1. 加载模型
2. 初始化tokenizer映射器(人名与类别)
3. 利用名字tokenizer将名字转换为索引,注意要padding
4. 用模型前向传播得到输出
5. 取出top3的索引
6. 将索引通过类别tokenizer转换为类别输出
'''

device=torch.device('cpu' if torch.backends.mps.is_available() else 'cpu')
#模型路径
model_path = 'char_rnn_classifier.pth'

# 1. 加载模型
net = torch.load(model_path, weights_only=False)
net.to(device)
# 2. 初始化tokenizer映射器(人名与类别)
name_tokenizer = tokenizer.get_letter_tokenizer()
cate_tokenizer = tokenizer.get_category_tokenizer()

def predict(name):
    net.eval() # 设置评估模式
    #3. 利用名字tokenizer将名字转换为索引,注意要padding
    name_idx_list = [0] * (19-len(name)) + name_tokenizer.encode(name) # python的基础列表相加是列表拼接的操作
    # 注意： 把列表转成成torch.tensor数据类型
    name_idx_tensor = torch.tensor(name_idx_list)
    with torch.no_grad(): # 禁用梯度
        # 4. 用模型前向传播得到输出
        pred = net(name_idx_tensor)
    # 5. 取出top3的索引
    value,idx = torch.topk(pred,k=3) #返回2个值，对应top k个向量的元素值，以及top k个向量的下标索引值
    # 6. 将索引通过类别tokenizer转换为类别输出
    cates = []
    for id in idx:
        cate = cate_tokenizer.idx2word[id.item()] # 注意把tensor转换标准的python数据类型
        cates.append(cate)
    return cates

if __name__ == '__main__':
    print(predict('Robin'))