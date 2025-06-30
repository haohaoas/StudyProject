import jieba.posseg as psg
r=psg.lcut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print(type(r[0]))
from jieba.posseg import pair
for w in psg.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造"):
    print(w.word,w.flag)