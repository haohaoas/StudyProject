# 导入必要的库
import joblib
import pandas as pd
'''
# 需求: 构建英语和法语的分词器并保存

# 改动步骤：
# 0. 导包，joblib和pandas. joblib安装：pip install joblib
# 1. Tokenizer中word2idx添加特殊符号<pad>, <sos>, <eos>,注意idx2word也要同步更新
# 2. Tokenizer中添加save和load方法，用于保存和加载分词器，load方法是静态方法
# 3. 编写build_tokenizer函数，接收句子列表和保存路径，创建Tokenizer实例，构建词表并保存
# 4. 在主程序中读取数据文件，提取句子列表，调用build_tokenizer函数创建分词器并保存
'''

class Tokenizer():

    def __init__(self, token_level: str = "word"):
        """
        Args:
            token_level: 分词级别，可选["word", "char"]
        """
        # 1. Tokenizer中word2idx添加特殊符号<pad>, <sos>, <eos>,注意idx2word也要同步更新
        self.word2idx = {'<pad>':0,'<sos>':1,'<eos>':2}  # 词到索引的映射
        self.idx2word = {id:token for token,id in self.word2idx.items()}  # 索引到词的映射
        self.token_level = token_level

    def _tokenize(self, text):
        """根据设置的分词级别和语言进行分词"""
        if self.token_level == "char":
            # 字符级别分词
            return list(text)
        else:
            return text.split()

    def build_vocab(self, sentences):
        """构建词汇表"""
        for sent in sentences:
            # 使用自定义的分词方法进行分词
            for word in self._tokenize(sent):
                if word not in self.word2idx:
                    # 遇到新词时扩展词典
                    self.word2idx[word] = len(self.word2idx)

        # 更新反向索引
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text):
        """将文本转换为索引序列"""
        tokens = self._tokenize(text)
        return [self.word2idx.get(token) for token in tokens]

    def decode(self, indices):
        """将索引序列还原为文本"""
        tokens = [self.idx2word.get(idx) for idx in indices]
        return " ".join(tokens)  # 英文用空格分隔

    def __len__(self) -> int:
        """获取词表大小"""
        return len(self.word2idx)

    # 2. Tokenizer中添加save和load方法，用于保存和加载分词器，load方法是静态方法
    def save(self,path):
        joblib.dump(self,path)

    @staticmethod
    def load(path):
        return joblib.load(path)


    @staticmethod
    # 把字符串转换为列表
    def str_to_list(list_string):
        # 需求1：此处构建一个把列表字符串转换为列表的函数，因为届时从文件中读取的[1,2,3]会是字符串
        # list_string = "[1, 2, 3, 4]" # string类型
        # 步骤1：去除方括号 .strip('[]')
        list_string = list_string.strip('[]')
        # 步骤2：分割字符串 .split(',')
        list_string = list_string.split(',')
        # 步骤3：转换为整数,列表表达式遍历转int即可
        return [int(i) for i in list_string]
        # 最后输出：[1, 2, 3, 4] # list类型

# 3. 编写build_tokenizer函数，接收句子列表和保存路径，创建Tokenizer实例，构建词表并保存
def build_tokenizer(sentences,path):
    tk = Tokenizer()
    tk.build_vocab(sentences)
    tk.save(path)



# 主程序（数据预处理流程）
if __name__ == "__main__":
    path = 'data/eng-fra-v2.txt'
    df = pd.read_csv(path,header=None,sep='\t')
    df.columns = ['en','fr']
    # 4. 在主程序中读取数据文件，提取句子列表，调用build_tokenizer函数创建分词器并保存
    # df[列名].to_list()可直接把那一列转成列表输出
    build_tokenizer(df['en'].to_list(), 'data/en_tokenizer.pkl')
    build_tokenizer(df['fr'].to_list(), 'data/fr_tokenizer.pkl')

