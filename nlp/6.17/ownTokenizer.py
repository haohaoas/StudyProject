import jieba


class Tokenizer():
    def __init__(self,token_level:str="word",lang:str="en"):#token_level 分词级别：word,char lang可选：en,zh
        self.word2idx  = {}#词向量
        self.idx2word = {}#索引向量
        self.token_level = token_level
        self.lang = lang

    def build_vocab(self, sentences):
        for sentence in sentences:
            if self.token_level == 'char':
                tokens = list(sentence.strip())
            elif self.token_level == 'word':
                if self.lang == 'zh':
                    tokens = list(jieba.cut(sentence))
                elif self.lang == 'en':
                    tokens = sentence.split()
                else:
                    raise ValueError("Unsupported language")
            else:
                raise ValueError("Unsupported token_level")
            for word in tokens:
                if word not in self.word2idx:
                    self.word2idx.setdefault(word, len(self.word2idx))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text):
        def tokenize(s):
            if self.token_level == 'char':
                return list(s)
            elif self.token_level == 'word':
                return s.split()
            else:
                raise ValueError("Unsupported token_level")
        if isinstance(text, list):  # 多条文本
            return [[self.word2idx.get(word) for word in tokenize(sentence)] for sentence in text]
        else:  # 单条文本
            return [self.word2idx.get(word) for word in text]
    def decode(self, text):
        if isinstance(text[0], list):  # 多条文本
            return [''.join([self.idx2word[idx] for idx in sentence]) for sentence in text]
        else:  # 单条文本
            return '' .join([self.idx2word[idx] for idx in text])
    def __len__(self):
        return len(self.word2idx)

def get_category_tokenizer():
    all_category = ['Czech','German','Arabic','Japanese','Chinese','Vietnamese','Russian',
                   'French','Irish','English','Spanish','Greek','Italian','Portuguese',
                   'Scottish','Dutch','Korean','Polish']
    cate_tokenizer = Tokenizer()
    cate_tokenizer.build_vocab(all_category)
    return cate_tokenizer

def get_letter_tokenizer():
    all_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ., ;'"
    name_tokenizer = Tokenizer('char')
    name_tokenizer.word2idx = {'<pad>':0}
    name_tokenizer.idx2word = {0:'<pad>'}
    name_tokenizer.build_vocab([all_letters])
    return name_tokenizer

if __name__ == '__main__':
    zh_sentences=['你好', '你不好']
    en_sentences=['hello', 'you are bad']
    tokenizer = Tokenizer(token_level='word',lang='en')
    tokenizer.build_vocab(en_sentences)
    print(tokenizer.word2idx)
    print(tokenizer.encode(en_sentences))
    print(tokenizer.decode(tokenizer.encode(en_sentences)))

