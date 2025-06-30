import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree,DecisionTreeRegressor
import matplotlib.pyplot as plt

# 泰坦尼克号存活预测
data = pd.read_csv('../5.13/train.csv')
data = data.drop(['PassengerId'], axis=1)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])  # one-hot编码
X = data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = DecisionTreeClassifier(criterion='gini')

model.fit(X_train_scaled, y_train)
print("模型准确率：", model.score(X_test_scaled, y_test))
print("模型特征重要性：", model.feature_importances_)
report = classification_report(y_test, model.predict(X_test_scaled), target_names=['Died', 'Survived'])
print(report)
plt.figure(figsize=(20,10),dpi=400)
plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['Died', 'Survived'])

plt.show()
