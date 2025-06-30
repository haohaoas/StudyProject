import joblib
import pandas as pd

try:
    import spacy
    nlp_fr = spacy.load('fr_core_news_sm')
except ImportError:
    nlp_fr = None  # 确保赋值，即使加载失败
class Tokenizer():
    def __init__(self,token_level:str="word",lang:str="en"):#token_level 分词级别：word,char lang可选：en,fr
        self.word2idx  = {}#词向量
        self.idx2word = {}#索引向量
        self.token_level = token_level
        self.lang = lang

    def _tokenize(self, sentence):
        if self.token_level == 'char':
            return list(sentence.strip())
        elif self.token_level == 'word':
            if self.lang == 'en':
                return sentence.strip().split()
            elif self.lang == 'fr':
                if nlp_fr:
                    return [token.text for token in nlp_fr(sentence)]
                else:
                    return sentence.strip().split()  # fallback 简单分词
            else:
                raise ValueError("Unsupported language")
        else:
            raise ValueError("Unsupported token_level")

    def build_vocab(self, sentences):
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            for word in tokens:
                if word not in self.word2idx:
                    self.word2idx.setdefault(word, len(self.word2idx))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text):
        if isinstance(text, list):  # 多条文本
            return [[self.word2idx.get(word) for word in self._tokenize(sentence)] for sentence in text]
        else:  # 单条文本
            return [self.word2idx.get(word) for word in self._tokenize(text)]
    def decode(self, text):
        if isinstance(text[0], list):  # 多条文本
            return [''.join([self.idx2word[idx] for idx in sentence]) for sentence in text]
        else:  # 单条文本
            return '' .join([self.idx2word[idx] for idx in text])
    # 把字符串转换为列表
    @staticmethod
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
    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        joblib.dump(self, path)
    def load(path):
        return joblib.load(path)

def build_tokenizer(sentences,path):
    tk = Tokenizer(token_level="char")  # 明确设定字符级分词
    tk.build_vocab(sentences)
    tk.save(path)
if __name__ == "__main__":
    path = 'data/eng-fra.txt'
    df = pd.read_csv(path,header=None,sep='\t')
    df.columns = ['en','fr']
    # 4. 在主程序中读取数据文件，提取句子列表，调用build_tokenizer函数创建分词器并保存
    # df[列名].to_list()可直接把那一列转成列表输出
    build_tokenizer(df['en'].to_list(), 'model/en_tokenizer.pkl')
    build_tokenizer(df['fr'].to_list(), 'model/fr_tokenizer.pkl')