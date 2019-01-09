import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """for ex5
    d['X'] shape = (12, 1)
    pandas has trouble taking this 2d ndarray to construct a dataframe, so I ravel
    the results
    """
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])

X, y, Xval, yval, Xtest, ytest = load_data()

# print X.shape,y.shape
# print Xval,X;
# df = pd.DataFrame({'water_level':X, 'flow':y})
# sns.lmplot('water_level', 'flow', data=df, fit_reg=False, height=7)
# plt.show()

X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

def cost(theta, X, y):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(n), linear regression parameters
    """
    m = X.shape[0]

    inner = X.dot(theta.T) - y  # R(m*1)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.dot(inner.T)
    cost = square_sum / (2 * m)

    return cost

theta = np.ones(X.shape[1])

#print cost(theta, X, y)

def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T.dot(X.dot(theta.T) - y)  # (m,n).T @ (m, 1) -> (n, 1)

    return inner / m
#print gradient(theta, X, y)

def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0  # don't regularize intercept theta

    regularized_term = (l / m) * regularized_term

    return gradient(theta, X, y) + regularized_term

#print regularized_gradient(theta, X, y)


def linear_regression_np(X, y, l=1):
    """linear regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res

def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()

    return cost(theta, X, y) + regularized_term

theta = np.ones(X.shape[0])

final_theta = linear_regression_np(X, y, l=0).get('x')

b = final_theta[0] # intercept
m = final_theta[1] # slope

# plt.scatter(X[:,1], y, label="Training data")
# plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
# plt.legend(loc=2)
# plt.show()


# training_cost, cv_cost = [], []

# m = X.shape[0]
# for i in range(1, m+1):
# #     print('i={}'.format(i))
#     res = linear_regression_np(X[:i, :], y[:i], l=0)
    
#     tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
#     cv = regularized_cost(res.x, Xval, yval, l=0)
# #     print('tc={}, cv={}'.format(tc, cv))
    
#     training_cost.append(tc)
#     cv_cost.append(cv)
# plt.plot(np.arange(1, m+1), training_cost, label='training cost')
# plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
# plt.legend(loc=1)
# plt.show()

def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.as_matrix() if as_ndarray else df
def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())

def prepare_poly_data(power,*args):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).as_matrix()

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]

X, y, Xval, yval, Xtest, ytest = load_data()
print poly_features(X, power=3)


X_poly, Xval_poly, Xtest_poly= prepare_poly_data(8, X, Xval, Xtest, )
print X_poly[:3, :]


