import jieba
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk  # 词性标注,ne_chunk#实体命名

# 英语分词
tree = ne_chunk(pos_tag(word_tokenize("Barack Obama was born in Hawaii.")))
print(tree)

# (S 句子节点标签
#   (PERSON Barack/NNP) 人名
#   (PERSON Obama/NNP) 人名
#   was/VBD 动词过去式
#   born/VBN 动词现在分词
#   in/IN  介词
#   (GPE Hawaii/NNP) 实体
#   ./.)

# 中文分词
import hanlp

hanlp_load = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)  # 微软亚洲研究院命名实体识别模型
# hanlp.pretrained.cws  # 分词模型
# hanlp.pretrained.ner  # 命名实体识别模型
# hanlp.pretrained.pos  # 词性标注模型
content = '我在北京清华大学读书。'
output = hanlp_load(content)
print(output)
