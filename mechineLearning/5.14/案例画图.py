from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
x=np.any(list(range(1,11))).reshape(-1,1)
x_test=np.arange(0.0,10.0,0.1).reshape(-1,1)
y=np.array([5.56,5.70,5.91,6.40,6.80,7.05,8.90,8.70,9.00,9.05])
model1=DecisionTreeRegressor(max_depth=1)
model2=DecisionTreeRegressor(max_depth=2)
model3=LinearRegression()
model1.fit(x,y)
model2.fit(x,y)
model3.fit(x,y)
y_pred1=model1.predict(x)
y_pred2=model2.predict(x)
y_pred3=model3.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred1,label='max_depth=1')
plt.plot(x,y_pred2,label='max_depth=2')
plt.plot(x,y_pred3,label='LinearRegression')
plt.show()