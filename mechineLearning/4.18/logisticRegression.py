import  numpy as np
import  pandas as pd
from  matplotlib import  rcParams
from sklearn.datasets import  load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

data=load_breast_cancer()
X=data.data
Y=data.target
print("数据集特征数量:",Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=4)
model=LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
print("准确率",accuracy_score(Y_test,Y_pred))
print("混淆矩阵",confusion_matrix(Y_test,Y_pred))
print("分类报告",classification_report(Y_test,Y_pred,target_names=data.target_names))