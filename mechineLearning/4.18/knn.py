import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris  # 加载鸢尾花数据
from sklearn.model_selection import train_test_split  # 划分训练和测试
from sklearn.neighbors import KNeighborsClassifier  # 机器学习算法：KNN
from sklearn.metrics import accuracy_score  # 用来算模型预测准确不准确
# iris=load_iris()
# X=iris.data
# Y=iris.target
# print('特征数据X的前五条')
# print(X[:5])
# print('标签数据Y的前五条')
# print(Y[:5])
# # 把数据随机分成训练集和测试集，测试集占20%
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# # print("训练集数量：", len(X_train))
# # print("测试集数量：", len(X_test))
#
# model=KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train, Y_train)
# y_pred=model.predict(X_test)
# print('预测结果：', y_pred)
# accuracy_score=accuracy_score(Y_test, y_pred)
# print('准确率：', Y_test)
# print('准确率：', accuracy_score)
# new_flower = [[5.9, 3.0, 5.1, 1.8]]
# prediction = model.predict(new_flower)
# print('预测结果：', prediction[0])
# target_names = iris.target_names
# print("预测的花的品种是：", target_names[prediction[0]])
# print(iris.feature_names)
# print('------'*30)
# # 几朵不同的新花样本
# test_flowers = [
#     [5.0, 3.4, 1.5, 0.2],  # 可能是 setosa
#     [6.0, 2.7, 5.1, 1.6],  # 可能是 virginica
#     [6.1, 2.8, 4.0, 1.3],  # 可能是 versicolor
# ]
#
# # 逐个预测
# for flower in test_flowers:
#     pred = model.predict([flower])[0]
#     print(f"花 {flower} 的预测结果是：{target_names[pred]}")
rcParams['font.sans-serif']=['SimHei']
rcParams['axes.unicode_minus']=False
# iris=load_iris()
# X=iris.data
# Y=iris.target
# target_names=iris.target_names
# x_Axis=X[:,2]
# y_Axis=X[:,3]
# for i,color in zip([0,1,2],['red','green','blue']):
#     plt.scatter(x_Axis[Y==i],y_Axis[Y==i],color=color,label=target_names[i])
# plt.xlabel('Petal length (cm)')
# plt.ylabel('Petal width (cm)')
# plt.title('莺尾花花瓣尺寸分布图')
# plt.grid(True)
# plt.show()
X = np.array([
    [1, 2], [2, 3], [3, 1],     # 红色点
    [6, 5], [7, 7], [8, 6]      # 蓝色点
])
y = np.array([0, 0, 0, 1, 1, 1])
new_point=np.array([[5,4]])
k=KNeighborsClassifier(n_neighbors=3)
k.fit(X,y)
pred=k.predict(new_point)
print(pred)
for i in range(len(X)):
    color='red' if y[i]==0 else 'blue'
    plt.scatter(X[i,0],X[i,1],color=color,s=100)
plt.scatter(new_point[0,0],new_point[0,1],color='green',s=150,marker='X',label='预测点')
distances=np.linalg.norm(X-new_point,axis=1)
nearest_indices=distances.argsort()[:3]
for i in nearest_indices:
    print(f'第{i+1}个最接近的点：{X[i]}')
for i in nearest_indices:
    plt.plot([X[i,0],new_point[0,0]],[X[i,1],new_point[0,1]],'k--')
plt.title(f'KNN预测类别：{"红色" if pred[0]==0 else "蓝色"}')
plt.legend()
plt.show()





