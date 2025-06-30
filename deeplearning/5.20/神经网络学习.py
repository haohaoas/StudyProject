#感知机
# def perceptron(x1,x2):
#     w1,w2,b=1,1,-1.5
#     z=w1*x1+w2*x2+b
#     return 1 if z>0 else 0
# print(perceptron(1,1))
# print(perceptron(0,1))
# print(perceptron(0,0))

#激活函数对比
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei']
# import numpy as np
# x=np.linspace(-10,10,100)
# sigmoid=1/(1+np.exp(-x))
# tanh=np.tanh(x)
# relu=np.maximum(x,0)
# plt.plot(x,sigmoid,label="sigmoid")
# plt.plot(x,tanh,label="tanh")
# plt.plot(x,relu,label="relu")
# plt.legend()
# plt.title("激活函数对比")
# plt.grid(True)
# plt.show()

#softmax学习
# import numpy as np
# z=np.array([2.0,1.0,0.1])
# y_true=np.array([0,1,0])
# exp_z=np.exp(z)
# softmax=exp_z/np.sum(exp_z)
# loss=-np.sum(y_true*np.log(softmax+1e-15))
# print(softmax)
# print(loss)
# 神经网络学习的步骤
# x=1.0
# y_true=10.0
# w=5.0 #初始权重
# lr=0.01#学习率
# for e in range(20):
#     y_pred=w*x#预测值
#     loss=(y_pred-y_true)**2#损失函数
#     grad=2*(y_pred-y_true)*x#梯度
#     w=w-lr*grad#更新权重
#     print(loss)
#     print(w)
