# 浅拷贝可变类型
# a=[1,2,3,4,5]
# b=[6,7,8,9,10,[1,2,3]]
# c=[11,12,13,14,15,a,b]
# print('-------'*30)
# print(id(a))
# print(id(b))
# print(id(c))
# print('--------'*30)
# d=c.copy()
# print(id(d))
# print(id(d[6]))

#浅拷贝不可变类型
import copy

# a=(1,2,3)
# b=(4,5,6,a)
# c=(7,8,9,a,b)
# print(id(a))
# print(id(b))
# print(id(c))
# print('------'*30)
# d=copy.copy(c)
# print(id(d))
# print(id(d[3]))
# print(id(d[4]))