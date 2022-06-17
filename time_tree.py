import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from scipy import linalg
import math
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import time

k_value = 20

def ridge_regression_train(X,y,lambda_value):
    X = np.array(X)
    XTy = np.matmul(X.transpose(),y)
    XTXl = np.matmul(X.transpose(),X)+lambda_value*np.identity(X.shape[1])
    alpha = np.matmul(linalg.pinv(XTXl), XTy)
    return alpha

def ridge_regression_predict(X,alpha):
    return np.matmul(X,alpha)

def find_error(pred_label, true_label):
    return math.sqrt(mean_squared_error(true_label, pred_label))


def reset(sample, label, n_train):
    sample_train = sample[0:n_train,:]
    sample_test = sample[n_train:,:]
    label_train = label[0:n_train]
    label_test = label[n_train:]  
    return sample_train, sample_test, label_train, label_test
        
# Process data
data = np.loadtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1] # first p-1 columns are features
label = data[:,-1] # last column is label
[n,p] = sample.shape

n_train = int(n*0.5)
sample = data[:,0:-1] # first p-1 columns are features
label = data[:,-1] # last column is label


sample_train = sample[0:n_train,:]

# scale data to standard gaussian
scaler = preprocessing.StandardScaler().fit(np.array(sample_train))
# scaled data
sample = scaler.transform(np.array(sample))

# 
print('Tree')
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
regr = DecisionTreeRegressor()
regr.fit(sample_train,label_train)
end_time = time.time()
tree_time = end_time-start_time
label_pred_test = regr.predict(sample_test)
mse_test = find_error(label_pred_test,label_test)

print("Random Forest")
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
regr = RandomForestRegressor(n_estimators=k_value)#max_depth=T, random_state=0)
regr.fit(sample_train,label_train)
end_time = time.time()
forest_time = end_time-start_time


print("Extra Tree")
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
reg = ExtraTreesRegressor(bootstrap=False,n_estimators=k_value, max_features=1).fit(sample_train,label_train)
end_time = time.time()
extra_time = end_time-start_time


print("starting rhsk")
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
hyps_train = []
for j in range(k_value):
    # print(i,j)
    # reg = ExtraTreesRegressor(bootstrap=True,n_estimators=1,max_depth=T, max_features=1).fit(sample_train,label_train)
    reg = ExtraTreesRegressor(bootstrap=True,n_estimators=1,max_features=1,max_samples=.8).fit(sample_train,label_train)
    label_pred_train = reg.predict(sample_train)
    hyps_train.append(label_pred_train)
hyps_train = np.array(hyps_train).transpose()
alpha = ridge_regression_train(hyps_train, label_train, 0)
end_time = time.time()
rhss_time = end_time-start_time

print('Showing Time')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
# langs = ['Decision Tree', 'Random Forest', 'Extra Tree', 'RHSS-Tree']
# times = [tree_time,forest_time,extra_time,rhss_time]
langs = ['Random Forest', 'Extra Tree', 'RHSS-Tree']
times = [forest_time,extra_time,rhss_time]
ax.bar(langs,times)
plt.show()

# =============================================================================
# data = [[30, 25, 50, 20],
# [40, 23, 51, 17],
# [35, 22, 45, 19]]
# X = np.arange(4)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
# =============================================================================
