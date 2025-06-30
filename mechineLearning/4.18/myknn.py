# 每个点的 [身高, 体重]
import math
from sklearn.datasets import load_iris
iris = load_iris()
X_train = iris.data
y_train = iris.target
x_test = [5.1, 3.5, 1.4, 0.2]
def knn_predict(x_test, X_train, y_train, k):
    distances = []
    for i in range(len(X_train)):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(x_test, X_train[i])))
        distances.append([d, y_train[i]])
    distances.sort()
    distances = distances[:k]
    count = sum(1 for d, label in distances if label == 1)
    count1 = k - count
    return '是运动员' if count > count1 else '不是运动员'
if __name__ == '__main__':
    distances = knn_predict(x_test, X_train, y_train, 3)
    print(distances)

