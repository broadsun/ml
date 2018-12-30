#coding=utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])

# print data.head()
# print data.describe()

def computeCost(X, y, theta):
    inner = np.power((X*theta.T-y), 2)
    return np.sum(inner)/(2*len(X))

data.insert(0, "Ones", 1)

cols = data.shape[1]

X = np.matrix(data)[:,:-1]
y = np.matrix(data)[:,-1]
# print X.shape,y.shape
theta = np.matrix(np.array([0,0]))

# print computeCost(X, y, theta)

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iters)
    for i in range(iters):
        error = X*theta.T - y
        temp = temp - ((alpha / len(X)) * np.sum(np.multiply(error, X), axis = 0))
        theta = temp
        cost[i] =  computeCost(X, y, theta)
    return theta,cost

import random
def randomgradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iters)
    for i in range(iters):
        dataIndex = range(X.shape[0])
        for j in range(X.shape[0]):
            alpha = 4/(1.0+i+j) + 0.0001
            idx = int(random.uniform(0, len(dataIndex)))
            error = X[idx,:]*theta.T - y[idx,:]
            #print error,float(error[0]);exit(0)
            temp = temp - (alpha * error * X[idx,:])
            theta = temp
            del(dataIndex[idx])
        cost[i] =  computeCost(X, y, theta)
        #print theta
    return theta, cost

# alpha = 0.01
# iters = 100
# theta = np.matrix(np.array([1,1]))
# g, cost = gradientDescent(X, y, theta, alpha, iters)
# theta = np.matrix(np.array([1,1]))
# g_r, cost_r = randomgradientDescent(X, y, theta, alpha, iters)

# print g,g_r


# x = np.linspace(data.Population.min(), data.Population.max(), 100)
# f = g[0, 0] + (g[0, 1] * x)
# f_r = g_r[0, 0] + (g_r[0, 1] * x)
# fig,ax = plt.subplots(figsize=(12,8))
# ax.plot(x, f, 'r', label='Prediction')
# ax.plot(x, f_r, 'b', label='Prediction')
# ax.scatter(data.Population, data.Profit, label = 'Training Set')
# ax.legend(loc = 2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()

# fig,ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(iters), cost, 'r')
# ax.plot(np.arange(iters), cost_r, 'b')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()

path =  'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print data2.head()
data2 = (data2 - data2.mean()) / data2.std()
print data2.head()

data2.insert(0, 'Ones', 1)

X2 = np.matrix(data2)[:,0:-1]
y2 = np.matrix(data2)[:,-1]
print X2.shape,y2.shape

theta2 = np.matrix(np.array([0,0,0]))
iters2 = 1000
alpha = 0.01
g2,cost2 = gradientDescent(X2, y2, theta2, alpha, iters2)

print g2,cost2

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters2), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


