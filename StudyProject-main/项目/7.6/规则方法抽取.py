import jieba.posseg as pseg

samples=["2014年1月8日，杨幂与刘恺威的婚礼在印度尼西亚巴厘岛举行",
           "周星驰和吴孟达在《逃学威龙》中合作出演",
           '成龙出演了《警察故事》等多部经典电影']
#定义需要抽取的关系集合
relation2dict={'夫妻关系':['结婚','领证','婚礼'],
         '合作关系':['合作','搭档','签约'],
         '演员关系':['出演','角色','主演'],
         }

for text in samples:
    entities=[]
    relations=[]
    move_name=[]
    for word,tag in pseg.lcut(text):
        if tag=='nr':
            entities.append(word)
        elif tag=='x':
            if len(move_name)==0:
                move_name.append(text.index(word))
            else:
                move_name.append(text.index(word))
                entities.append(text[move_name[0]+1:move_name[1]])
        else:
            for key,value in relation2dict.items():
                if word in value:
                    relations.append(key)

    if len(entities)>=2 and len(relations)>=1:
        print('text',text)
        print('提取结果：', entities[0] + '->' + relations[0] + '->' + entities[1])
    else:
        print("原始文本：", text)
        print('不好意思，暂时没能从文本中提取出关系结果')
    print('*' * 80)