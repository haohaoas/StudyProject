import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#逻辑回归初步使用
# X = np.array([
#     [180, 80],
#     [160, 50],
#     [170, 65],
#     [155, 48],
#     [165, 55]
# ])
# y = np.array([1, 0, 1, 0, 0])
# scaler = StandardScaler()
# X_scaled=scaler.fit_transform(X)
# X_test=np.array([[172, 70]])
# X_test_scaled=scaler.transform(X_test)
# model = LogisticRegression()
# model.fit(X_scaled, y)
# print(f'预测概率：{model.predict_proba(X_test_scaled)}')
#
# print(f'预测结果：{model.predict(X_test_scaled)}')
#计算准确率等
# y_true = [1,1,0,1,0,0,1,0,1,0]
# y_pred = [1,0,0,1,0,1,1,0,1,1]
# print(f'混淆矩阵：{confusion_matrix(y_true, y_pred)}')
# print(accuracy_score(y_true, y_pred))#准确率 所有的预测正确的结果占所有结果比例
# print(precision_score(y_true, y_pred))#精准率 预测正确的结果占所有预测的结果比例
# print(recall_score(y_true, y_pred))#召回率 所有预测的结果中，被模型预测正确的结果占所有预测的结果比例
# print(f1_score(y_true, y_pred))#f1值 精准率和召回率的 调和 平均值
# report=classification_report(y_true, y_pred,target_names=['女','男'])#输出报告
# print(report)#accuracy 总体准确率：预测正确的比例 macroavg 所有类别的平均值：不考虑样本数量 weighted avg 所有样本的加权重平均值 :更真实反映模型综合表现

#计算缺失值
# data = pd.DataFrame({
#     'Age': [25, np.nan, 40, 30],
#     'Income': [8000, 10000, np.nan, 9000],
#     'Gender': ['男', '女', '女', np.nan]
# })
# print(f'原始数据：\n{data}\n')
# data['Age']=data['Age'].fillna(data['Age'].mean())
# data['Gender']=data['Gender'].fillna(data['Gender'].mode()[0])#取出现次数最多的数
# print(f'填充后的数据：\n{data}')
#泰坦尼克号存活预测
# titanic_data=pd.read_csv('train.csv')
# titanic_data=titanic_data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]#删除无用列 SibSp和Parch为兄弟姐妹和父母数量，Fare为船票价格，Embarked为上船地点
# titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())
# titanic_data['Embarked']=titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
# data=pd.get_dummies(titanic_data,columns=['Sex','Embarked'])#one-hot编码
# X=data.drop(columns=['Survived'])
# y=data['Survived']
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# scaler=StandardScaler()
# X_train_scaler=scaler.fit_transform(X_train)
# X_test_scaler=scaler.transform(X_test)
# model=LogisticRegression()
# model.fit(X_train_scaler,y_train)
# print("模型准确率：", model.score(X_test_scaler,y_test))
# classification_report=classification_report(y_test,model.predict(X_test_scaler))#模型报告
# print(classification_report)

#癌症数据集测试
cancer = load_breast_cancer()
X=cancer.data
y=cancer.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)
model=LogisticRegression()
model.fit(X_train_scaler,y_train)
print("模型准确率：", model.score(X_test_scaler,y_test))

#电信用户流失预测
telco_data=pd.read_csv('churn.csv')

X=telco_data.drop(columns=['Churn'])
X=pd.get_dummies(X)
y=telco_data['Churn'].map({'No':0,'Yes':1})
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=LogisticRegression()
model.fit(X_train,y_train)
print("模型准确率：", model.score(X_test,y_test))
print("模型预测结果：", model.predict(X_test))
y_prob=model.predict_proba(X_test)[:,1]
threshold=0.3
y_pred=np.where(y_prob>threshold,1,0)
report = classification_report(y_test, model.predict(X_test))
tpr,fpr,thresholds=roc_curve(y_test,y_prob)
auc=roc_auc_score(y_test,y_prob)
s = classification_report(y_test, y_pred)
print(s)
plt.figure()
plt.plot(fpr,tpr, label=f'ROC 曲线 (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.legend()
plt.show()
#测试正则化强度参数C
#l2
# model=LogisticRegression(penalty='l2',solver='liblinear',C=1)
# model.fit(X_train, y_train)
# coef_=model.coef_[0]
# feature_names=X.columns
# for feature,coef in zip(feature_names,coef_):
#     print(f'特征名：{feature},系数：{coef}')
#l1
# model=LogisticRegression(penalty='l1',solver='liblinear',C=1)
# model.fit(X_train, y_train)
# print(model.coef_)



