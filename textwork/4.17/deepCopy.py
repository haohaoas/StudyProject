import copy
#深拷贝可变类型
# if __name__ == '__main__':
#     a = [1,2,3]
#     b=[11,12,33]
#     c=[4,5,6,a,b]
#     print(id(a))
#     print(id(c))
#     print('-----'*30)
#     d=copy.deepcopy(c)
#     print(id(d))
#     print(id(d[3]))
#     print('-----'*30)
#     a[1]=99
#     print(id(a))
#     print(id(d[3]))

# 深拷贝不可变类型
# if __name__ == '__main__':
#     a = (1,2,3)
#     b=(4,5,6,a)
#     c=(7,8,9,a,b)
#     print(id(a))
#     print(id(c))
#     print('------'*30)
#     d=copy.deepcopy(c)
#     print(id(d))
#     print(id(d[3]))
#     print('------'*30)
#     a=(11,22,33)
#     print(id(a))
#     print(id(d[3]))
# if __name__ == '__main__':
#     a =(1,'2',3)
#     b=[11,12,33]
#     c=(4,5,6,a,b)
#     print(id(a))
#     print(id(c))
#     print('-----'*30)
#     d=copy.deepcopy(c)
#     print(id(d))