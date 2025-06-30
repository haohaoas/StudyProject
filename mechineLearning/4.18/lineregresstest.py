import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x=[[80,86],[82,80],[85,78],[90,90],[86,82],[82,90],[78,80]]
y=[84.2,80.6,80.1,90,83.2,87.6,79.4]
x1=[xi[0] for xi in x]
plt.scatter(x1,y)
plt.show()
model=LinearRegression()
model.fit(x,y)
print(model.coef_)#权重
print(model.intercept_)#截距
print(model.predict([[90,80]]))
print(train_test_split)
