import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
speed = [86,87,88,86,87,85,86]
print(np.median(speed))#中位数 将数据排序后处于中间的数，如果有两个则是他们相加处以2
print(np.std(speed))#标准差 数据的平均值/2
print(np.var(speed))#方差 每个数据减去平均值再平方的和的平均值
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
print(np.percentile(ages,75))#百分位数 百分之多少的人小于多少岁
x=np.random.uniform(0.0,0.5,100000)#创建介于0和0.5之间250个随机数 指定平均值为0.0  和标准差为0.5
plt.hist(x,1000)
plt.show()
x=np.random.normal(5.0,1.0,100000)#创建介于5和1.0之间100000个随机数 指定平均值为5.0  和标准差为1
plt.hist(x,100)
plt.show()
x=np.random.normal(5.0,1.0,1000)
y=np.random.normal(10.0,2.0,1000)
plt.scatter(x, y)
plt.show()
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x, y)
plt.show()