#coding=utf8
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
# plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
# plt.scatter(df[100:150]['sepal length'], df[100:150]['sepal width'], label='2')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()


data = np.array(df.iloc[:100, [0, 1, -1]])

X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])
print data.shape,X.shape
# 数据线性可分，二分类数据
# 此处为一元一次线性方程
class Model:
    def __init__(self):
        self.w = np.ones(X.shape[1], dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        # self.data = data
        print self.w.shape
    
    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        if y >=0:
            return 1
        else:
            return -1
    
    # 随机梯度下降法
    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate*np.dot(y, X)
                    self.b = self.b + self.l_rate*y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'
        
    def score(self):
        pass

perceptron = Model()
perceptron.fit(X, y)

x_points = np.linspace(4, 7,10)
y_ = -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]

plt.plot(x_points, y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# from sklearn.linear_model import Perceptron
# clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
# clf.fit(X, y)

# print clf.coef_
# print clf.intercept_
# x_ponits = np.arange(4, 8)
# print x_ponits
# y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
# plt.plot(x_ponits, y_)

# plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
# plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()