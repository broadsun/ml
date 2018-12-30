#coding=utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex2data1.txt'

data = pd.read_csv(path, header =None, names=['Exam 1', 'Exam 2', 'Admitted'])
# print data.head()

# positive = data[data['Admitted'].isin([1])]
# negative = data[data['Admitted'].isin([0])]

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

# nums = np.arange(-10, 10, step = 1)

# # fig,ax = plt.subplots(figsize = (12,8))
# # ax.plot(nums, sigmoid(nums), 'r')
# # plt.show()

# def cost(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
#     second = np.multiply(1-y, np.log(1-sigmoid(X* theta.T)))
#     return np.sum(first-second)/len(X)


# data.insert(0, 'Ones', 1)

# X = np.matrix(data)[:,0:-1]
# y = np.matrix(data)[:,-1]
# theta = np.zeros(X.shape[1])
# print X.shape, y.shape, theta.shape

# def gradient(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     error = sigmoid(X * theta.T) - y
#     grad = np.sum(np.multiply(error, X), axis = 0)/len(X)
#     return grad

# # print gradient(theta, X, y)
# import scipy.optimize as opt
# result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
# #print result
# print cost(result[0], X, y)

# def predict(theta, X):
#     probability = sigmoid(X * theta.T)
#     return [1 if x >= 0.5 else 0 for x in probability]

# theta_min = np.matrix(result[0])
# print theta_min
# predictions = predict(theta_min, X)
# correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
# print sum(correct)


path =  'ex2data2.txt'
data2 = pd.read_csv(path, header = None, names = ['Test 1', 'Test 2', 'Accepted'])
print data2.head()

fig,ax = plt.subplots(figsize = (12,8))
positive = data2[data2['Accepted']==1]
negative = data2[data2['Accepted']==0]
ax.scatter(positive['Test 1'], positive['Test 2'])
ax.scatter(negative['Test 1'], negative['Test 2'])
plt.show()

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']
data2.insert(3,'Ones',1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

print data2.head()
