import jieba

content='传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
print(jieba.cut( content))
print(jieba.cut( content,cut_all=True))
print(jieba.cut_for_search( content))
print(jieba.lcut( content))
jieba.load_userdict("dict.txt")
word=jieba.cut( content)
print(list(word))