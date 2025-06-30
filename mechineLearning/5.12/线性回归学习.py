import matplotlib.pyplot as plt
import numpy as np
from LinearRegression import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt
import scipy as stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# y = [56.3, 60.6, 65.1, 68.5, 75]
# x = [[160], [166], [172], [174], [180]]
# model = LinearRegression()
# model.fit(x, y)
# print(model.predict([[176]]))
# print(model.coef_)  # 斜率
# print(model.intercept_)  # 截距
# # X = np.vstack([np.ones_like(x), x])  # 创建一个矩阵
# # print(X)
# x = np.array(x)
# w = np.linalg.inv(x.T @ x) @ x.T @ y  # 矩阵的转置乘以矩阵
# print(w)
X = np.array([
    [60, 1],
    [75, 2],
    [80, 3],
    [90, 5],
    [65, 8],
    [85, 6],
    [70, 9],
    [100, 10],
    [95, 4],
    [60, 12]
])
y = np.array([230.5, 276, 291.5, 322.5, 249, 308, 264.5, 355, 337, 236])
model=LinearRegression()
model.fit(X,y)
# print(model.coef_[0])#系数ß1
# print(model.coef_[1])#系数ß2
# print(model.intercept_)# 截距 ß0
# print(r2_score(y,model.predict(X)))#R2值
# print(mean_squared_error(y,model.predict(X)))#MSE
# rid=Ridge(alpha=1.0)
# rid.fit(X,y)
# y_ridge=rid.predict(X)
# print("【Ridge】面积系数：", rid.coef_[0])
# print("【Ridge】楼层系数：", rid.coef_[1])
# print("【Ridge】MSE：", mean_squared_error(y, y_ridge))
# lasso=Lasso(alpha=1.0)
# lasso.fit(X,y)
# y_lasso=lasso.predict(X)
# print("【Lasso】面积系数：", lasso.coef_[0])
# print("【Lasso】楼层系数：", lasso.coef_[1])
# print("【Lasso】MSE：", mean_squared_error(y, y_lasso))
#正规方程手写
# X = np.array([
#     [1, 60, 1],
#     [1, 75, 2],
#     [1, 80, 3],
#     [1, 90, 5],
#     [1, 65, 8],
#     [1, 85, 6],
#     [1, 70, 9],
#     [1, 100, 10],
#     [1, 95, 4],
#     [1, 60, 12]
# ])
# y = np.array([230.5, 276, 291.5, 322.5, 249, 308, 264.5, 355, 337, 236])
# beta=np.linalg.inv(X.T@X)@X.T@y
# print(f'正规方程系数：{beta}')
# 梯度下降手写
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X)
# model=SGDRegressor(max_iter=1000,learning_rate='constant',eta0=0.01)#创建模型 max_iter是迭代次数，learning_rate控制步长；
# model.fit(X_train, y)
# print("截距 β0：", model.intercept_[0])
# print("系数 β：", model.coef_)
# print("MSE：", mean_squared_error(y, model.predict(X_train)))
#预测boston房价
boston_data=pd.read_csv('../5.9/boston.csv')
x = boston_data.drop(columns=['MEDV'])  # 所有特征
y = boston_data['MEDV']                 # 房价
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2324235)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
# 4. 模型评估
print("R² 分数：", r2_score(Y_test, Y_pred))
print("MSE：", mean_squared_error(Y_test, Y_pred))
# 5. 系数解释（特征重要性）
coef_df = pd.DataFrame({
    'Feature': x.columns,
    'Coefficient': model.coef_
})
print(coef_df.sort_values(by='Coefficient', ascending=False))