from collections import Counter

from LinearRegression import r2_score
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import calinski_harabasz_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import rcParams
import pandas as pd
import joblib
import cv2
import seaborn as sns
from matplotlib import pyplot as plt

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# x=[[39,0,31],[3,2,65],[2,3,55],[9,38,2],[8,34,17],[5,2,57],[21,17,5],[45,2,9]]
# y=[0,1,2,2,2,1,0,0]
# clf=KNeighborsClassifier(n_neighbors=3)
# clf.fit(x,y)
# print(clf.predict([[23,3,17]]))
# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(X, y)
# print(clf.predict([[4]]))
# 归一化
# data = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]
# scaler = preprocessing.MinMaxScaler()
# data_scaled = scaler.fit_transform(data)
# print(data_scaled)
# 特征预处理
# data = [[90, 2, 10, 40],
# [60, 4, 15, 45],
# [75, 3, 13, 46]]
# scaler =StandardScaler()
# data_scaled = scaler.fit_transform(data)
# print(data_scaled)
# print(data_scaled.mean(axis=0))#均值
# print(data_scaled.std(axis=0))#标准差
#
# x,y=make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=
#                [0.4,0.2,0.2,0.2],random_state=22)
# plt.figure()
# plt.scatter(x[:,0],x[:,1],marker='o')
# plt.show()
# y_pred=KMeans(n_clusters=2,random_state=22,init='k-means++').fit_predict(x)
# plt.scatter(x[:,0],x[:,1],c=y_pred)
# plt.show()
# print('1-->',calinski_harabasz_score(x,y_pred))
# y_pred=KMeans(n_clusters=3,random_state=22).fit_predict(x)
# plt.scatter(x[:,0],x[:,1],c=y_pred)
# plt.show()
# print('2-->',calinski_harabasz_score(x,y_pred))
# y_pred=KMeans(n_clusters=4,random_state=22).fit_predict(x)
# plt.scatter(x[:,0],x[:,1],c=y_pred)
# plt.show()
# print('3-->',calinski_harabasz_score(x,y_pred))
# 显示莺尾花数据
# iris = load_iris()
# for i in range(1000):
#     seed=i
#     X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=seed)
#     iris_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
#     iris_data['target'] = iris.target
#     standard_scaler = StandardScaler()
#     X_train=standard_scaler.fit_transform(X_train)
#     X_test=standard_scaler.transform(X_test)
# # x_col = 'petal length (cm)'
# # y_col = 'petal width (cm)'
# # sns.lmplot(x=x_col, y=y_col, data=iris_data, hue='target', fit_reg=False)
# # plt.xlabel(x_col)
# # plt.ylabel(y_col)
# # plt.show()
#     model=KNeighborsClassifier(n_neighbors=3)
#     # model=GridSearchCV(model,param_grid={'n_neighbors':[1,3,5,7]},cv=5)
#     model.fit(X_train, Y_train)
#     model.predict(X_test)
#     print(accuracy_score(Y_test, model.predict(X_test)))
#     print(model.score(X_test, Y_test))
# my_csv=pd.DataFrame(model.cv_results_)
# my_csv.to_csv('result.csv',index=False)
# knn实现扫描手写数字
def show_digit(idx):
    data = pd.read_csv('../5.9/手写数字识别.csv')
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)

    param_grid = {'n_neighbors': [1, 3, 5, 7]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train_scaled, Y_train)

    model = grid_search.best_estimator_
    # joblib.dump(model, '../5.9/model.pkl')
    # joblib.dump(min_max_scaler, '../5.9/scaler.pkl')

    y_pred = model.predict(X_test_scaled)
    print(f'准确率：{accuracy_score(Y_test, y_pred)}')
    print('预测结果：', y_pred)
    print(grid_search.best_score_)
    print(grid_search.cv_results_)
    # data_ = x.iloc[idx].values
    # data_ = data_.reshape(28, 28)
    # plt.imshow(data_, cmap='gray')
    # plt.show()
if __name__ == '__main__':
    show_digit(1)
# if __name__ == '__main__':
#     show_digit(1)
# #测验手写图片数字
# photo = cv2.imread('../5.9/demo.png', cv2.IMREAD_GRAYSCALE)
# plt.imshow(photo, cmap='gray')
# plt.title("输入图像")
# plt.show()
# photo = photo.reshape(1, -1)
# scaler = joblib.load('../5.9/scaler.pkl')
# columns = [f'pixel{i}' for i in range(784)]
# photo_df = pd.DataFrame(photo, columns=columns)
# photo = scaler.transform(photo_df)
# knnModel = joblib.load('../5.9/model.pkl')
# y_pred = knnModel.predict(photo)
# proba = knnModel.predict_proba(photo)
# print(f'您预测的数字是：{y_pred[0]}')
# print(f'预测的概率分布为：{proba}')

# knn回归方法测试
# X = [[0, 0, 1], [1, 1, 0], [3, 10, 10], [4, 11, 12]]
# y = [0.1, 0.2, 0.3, 0.4]
# model=KNeighborsRegressor(n_neighbors=2)
# model.fit(X, y)
# y_pred=model.predict(X)
# print('预测结果：', y_pred)
# print('准确率：', model.score(X, y))
# print('准确率：', r2_score(y, y_pred))
# print(model.predict([[3,11,10]]))
