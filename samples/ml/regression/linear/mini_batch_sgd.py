import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()

data.describe()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()


def  computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列
X.head()#head()是观察前5行
y.head()

X = np.matrix((X.values))
y = np.matrix((y.values))
theta = np.matrix(np.array([0,0]))

computeCost(X, y, theta)


def miniBatchGradientDescent(X, y, theta, alpha, iters,batch_size):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    train_indexs = [i for i in range(len(X))]
    #print('train_indexs',train_indexs)
    random.shuffle(train_indexs)
    #print('train_indexs after', train_indexs)


    for i in range(iters):
        # error = (X * theta.T) - y  batch
        sample_index = i % (len(X)//batch_size)
        start = sample_index*batch_size
        end = sample_index*batch_size+batch_size
        print('start end are ......',start,end)
        #print('sample index ', sample_index, X.shape)
        error = (X[start:end, :]*theta.T)-y[start:end]
        #error = (X[sample_index, :] * theta.T) - y[sample_index]

        for j in range(parameters):
            # term = np.multiply(error, X[:, j])
            term = np.multiply(error, X[start:end, j])
            #term = np.multiply(error, X[sample_index, j])
            #print('term is ', term)
            #temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
            temp[0, j] = theta[0, j] - (alpha / batch_size)* np.sum(term)

        theta = temp
        cost[i] = computeCost(X, y, theta)

        # if sample_index == len(X)//batch_size -1:
        #     random.shuffle(train_indexs)

    return theta, cost

alpha = 0.01
iters = 1000

start = time.time()
g, cost = miniBatchGradientDescent(X, y, theta, alpha, iters, 96)
end = time.time()
print('cost time is:',end-start)

computeCost(X, y, g)


x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()