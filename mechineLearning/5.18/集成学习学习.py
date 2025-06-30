from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix, \
    ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 红酒品质分类预测
# data_train = pd.read_csv('红酒品质分类_train.csv')
# data_train=data_train[data_train['quality']!=1]
# data_test=pd.read_csv('红酒品质分类_test.csv')
# data_test=data_test[data_test['quality']!=1]
# def map_quality(q):
#     if q <= 2:
#         return 0  # 差
#     elif q <= 3:
#         return 1  # 中
#     else:
#         return 2  # 好
#
# data_train['quality'] = data_train['quality'].apply(map_quality)
# data_test['quality'] = data_test['quality'].apply(map_quality)
# X_train=data_train.iloc[:,:-1]
# y_train=data_train.iloc[:,-1]
# X_test=data_test.iloc[:,:-1]
# y_test=data_test.iloc[:,-1]
# encoder = LabelEncoder()
# y_train=encoder.fit_transform(y_train)
# y_test=encoder.transform(y_test)
# tree_model = DecisionTreeClassifier(criterion='gini',max_depth=2)
# tree_model.fit(X_train,y_train)
# print(classification_report(y_test,tree_model.predict(X_test)))
# adaboost_model = AdaBoostClassifier(estimator=tree_model,n_estimators=300,learning_rate=0.1)
# adaboost_model.fit(X_train,y_train)
# print(classification_report(y_test,adaboost_model.predict(X_test)))

# 泰坦尼克号存活预测(使用GBDT)
# data = pd.read_csv('../5.13/train.csv')
# data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# data['Age'] = data['Age'].fillna(data['Age'].mean())
# X = data.drop(columns=['Survived'])
# y = data['Survived']
# X = pd.get_dummies(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# estimator = GradientBoostingClassifier()
# model = GridSearchCV(estimator, param_grid={'n_estimators': [100, 110, 120, 130], 'max_depth': [2, 3, 4],
#                                             'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5], 'random_state': [9]}, cv=5)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(model.best_estimator_)

#红酒品质划分（使用GBDT）
data_train=pd.read_csv('红酒品质分类_train.csv')
data_test=pd.read_csv('红酒品质分类_test.csv')
def map_quality(q):
    if q <= 2:
        return 0  # 差
    elif q <= 3:
        return 1  # 中
    else:
        return 2  # 好

data_train['quality'] = data_train['quality'].apply(map_quality)
data_test['quality'] = data_test['quality'].apply(map_quality)
X_train = data_train.iloc[:, :-1]
X_test = data_test.iloc[:, :-1]
y_train=data_train.iloc[:,-1]
y_test=data_test.iloc[:,-1]
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
gbdt_model=GradientBoostingClassifier(random_state=22,learning_rate=0.1,n_estimators=100,max_depth=3)
gbdt_model.fit(X_train,y_train)
adb_model=AdaBoostClassifier(random_state=22,learning_rate=0.1,n_estimators=100)
adb_model.fit(X_train,y_train)
print(f'分类指标：{classification_report(y_test,adb_model.predict(X_test))}')
print(f'分类指标{classification_report(y_test,gbdt_model.predict(X_test))}')
matrix = confusion_matrix(y_test, gbdt_model.predict(X_test))
disp=ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
#显示模型特征重要性
importances = gbdt_model.feature_importances_
features = data_train.columns[:-1]
indices = np.argsort(importances)
plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature Importances (GBDT)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
