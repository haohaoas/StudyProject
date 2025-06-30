import fasttext
model=fasttext.train_supervised(
    '/Users/haohao/PycharmProjects/pythonProject/nlp/6.29/data/cooking.pre.train',
    lr=0.001,
    epoch=10,
    wordNgrams=2,
    autotuneValidationFile='/Users/haohao/PycharmProjects/pythonProject/nlp/6.29/data/cooking.pre.valid',
    loss='softmax',
    autotuneDuration=10,
                          )
result=model.test('/Users/haohao/PycharmProjects/pythonProject/nlp/6.29/data/cooking.pre.valid')
print(  result[0])
print(  result[1])
print(  result[2])