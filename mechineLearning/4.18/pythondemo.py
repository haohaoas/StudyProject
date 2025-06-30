import  numpy as np
import  matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
# rcParams['font.sans-serif']=['SimHei']
# rcParams['axes.unicode_minus']=False
# 面积=np.array([50,100,150,200,250])
# 房价=np.array([150000,300000,450000,600000,750000])
#
# x_mean=np.mean(面积)
# y_mean=np.mean(房价)
# numberator=np.sum((面积-x_mean)*(房价-y_mean))
# denominator=np.sum((面积-x_mean)**2)
# w=numberator/denominator
# b=y_mean-w*x_mean
# def predict(x):
#     return w*x+b
# 预测值=predict(200)
# print(f'预测面积为200的房价为{预测值}')
# plt.scatter(面积,房价,color='red',label='真实数据')
# plt.plot(面积,predict(面积),color='blue',label='拟合曲线')
# plt.xlabel('面积')
# plt.ylabel('房价')
# plt.legend()
# plt.title('面积与房价的关系')
# plt.show()
# 真实房价 = np.array([150000, 300000, 450000, 600000, 750000])
#
# # 预测房价数据（使用你的线性回归模型计算）
# 预测房价 = np.array([155000, 290000, 460000, 590000, 740000])
# MSE=np.sum((真实房价-预测房价)**2)
# print(f'均方误差为{MSE}')
面积 = np.array([50, 100, 150, 200, 250]).reshape(-1, 1)
房价 = np.array([150000, 300000, 450000, 600000, 750000]).reshape(-1, 1)
scaler = MinMaxScaler()
面积归一化=scaler.fit_transform(面积)
房价归一化=scaler.fit_transform(房价)
print("归一化后的面积数据:", 面积归一化.flatten())
print("归一化后的房价数据:", 房价归一化.flatten())
x_mean=np.mean(面积归一化)
y_mean=np.mean(房价归一化)
numberator=np.sum((面积归一化-x_mean)*(房价归一化-y_mean))
denominator=np.sum((面积归一化-x_mean)**2)
w=numberator/denominator
b=y_mean-w*x_mean
def predict(x):
    return w*x+b
预测值=predict(0.64)
预测房价_实际 = scaler.inverse_transform([[预测值]])[0][0]
print(f'预测面积为 160 的实际房价为: {预测房价_实际}')
MSE = np.mean((预测房价_实际 - scaler.inverse_transform(房价归一化)) ** 2)
print(f'均方误差 (MSE): {MSE}')
RMSS=np.sqrt(MSE)
print(f'均方根误差 (RMSE): {RMSS}')


