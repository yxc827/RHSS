import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import statistics as stat
from scipy import linalg
import math
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib


n_rounds = 20
k_values = [1,3,6,12,24,48,60,80,100]
gamma_value = 0.01
T = None

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
data = np.loadtxt('adult.csv', delimiter=',')
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


# Baseline (Decision Tree)
print('Standard Tree')
error_test = []
sum_test = 0
for i in range(n_rounds):
    # print(i)
    # reset variables
    sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
    
    # regr = DecisionTreeRegressor(max_depth=T)
    regr = DecisionTreeRegressor()
    regr.fit(sample_train,label_train)
    label_pred_test = regr.predict(sample_test)
    mse_test = find_error(label_pred_test,label_test)
    # store error in list
    error_test.append(mse_test)
    # compute sum of errors
    sum_test += mse_test
err_dt = [sum_test/n_rounds for _ in range(len(k_values))]
std_dt = [stat.stdev(error_test) for _ in range(len(k_values))]

# Extra Trees + RHS
print('RHSS-Tree')
err_rvfl_rhs = []
std_rvfl_rhs = []
for k_value in k_values:
    sum_test = 0
    error_test = []
    for i in range(n_rounds):
        # reset variables
        sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
        hyps_test = []
        hyps_train = []
        matrix_tests = []
        for j in range(k_value):
            # print(i,j)
            # reg = ExtraTreesRegressor(bootstrap=True,n_estimators=1,max_depth=T, max_features=1).fit(sample_train,label_train)
            reg = ExtraTreesRegressor(bootstrap=True,n_estimators=1,max_features=1,max_samples=.8).fit(sample_train,label_train)
            label_pred_test = reg.predict(sample_test)
            hyps_test.append(label_pred_test)
            label_pred_train = reg.predict(sample_train)
            hyps_train.append(label_pred_train)
        hyps_train = np.array(hyps_train).transpose()
        hyps_test = np.array(hyps_test).transpose()
        
        # Find alpha
        alpha = ridge_regression_train(hyps_train, label_train, 0)
        label_pred_test = ridge_regression_predict(hyps_test, alpha)
        mse_test = find_error(label_pred_test,label_test)
        # store error in list
        error_test.append(mse_test)
        # compute sum of errors
        sum_test += mse_test
    err_rvfl_rhs.append(sum_test/n_rounds)
    std_rvfl_rhs.append(stat.stdev(error_test))

# Extra Trees Regressor
print('Extra Tree')
avg_test_err_extra = []
stdev_test_extra = []
for k_value in k_values:
    sum_test = 0
    error_test = []
    for i in range(n_rounds):
        # print(i)
        # reset variables
        sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
        # reg = ExtraTreesRegressor(bootstrap=False,n_estimators=k_value, max_features=1,max_depth=T).fit(sample_train,label_train)
        reg = ExtraTreesRegressor(bootstrap=False,n_estimators=k_value, max_features=1).fit(sample_train,label_train)
        label_pred_test = reg.predict(sample_test)
        mse_test = find_error(label_pred_test,label_test)
        # store error in list
        error_test.append(mse_test)
        # compute sum of errors
        sum_test += mse_test
    avg_test_err_extra.append(sum_test/n_rounds)
    stdev_test_extra.append(stat.stdev(error_test))

# (Random Forest Regressor)
print('Random Forest')
err_base= []
std_base= []
for k_value in k_values:

    error_test = []
    sum_test = 0
    for i in range(n_rounds):
        # print(i)
        # reset variables
        sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
        
        regr = RandomForestRegressor(n_estimators=k_value)#max_depth=T, random_state=0)
        regr.fit(sample_train,label_train)
        label_pred_test = regr.predict(sample_test)
        mse_test = find_error(label_pred_test,label_test)
        # store error in list
        error_test.append(mse_test)
        # compute sum of errors
        sum_test += mse_test
    err_base.append(sum_test/n_rounds)
    std_base.append(stat.stdev(error_test))

plt.xlabel('Value of k', fontsize=18)
plt.ylabel('Testing RMSE', fontsize=18)
plt.plot(k_values, err_dt, label="Decision Tree",color='g',linestyle='--')
plt.plot(k_values, err_base, label="Random Forest",color='y',linestyle='--')
plt.plot(k_values, avg_test_err_extra, label="Extra Tree",color='C0',linestyle='--')
plt.plot(k_values, err_rvfl_rhs,color='darkorange', label="RHSS-Tree")
plt.fill_between(k_values, np.array(err_dt)-np.array(std_dt),np.array(err_dt)+np.array(std_dt),color='g', alpha=0.2)
plt.fill_between(k_values, np.array(err_base)-np.array(std_base),np.array(err_base)+np.array(std_base), color='y',alpha=0.2)
plt.fill_between(k_values, np.array(avg_test_err_extra)-np.array(stdev_test_extra),np.array(avg_test_err_extra)+np.array(stdev_test_extra),color='C0', alpha=0.2)
plt.fill_between(k_values, np.array(err_rvfl_rhs)-np.array(std_rvfl_rhs),np.array(err_rvfl_rhs)+np.array(std_rvfl_rhs),color='darkorange',alpha=0.2)
plt.legend(loc="upper right")
matplotlib.rc('font', size=18)
plt.xscale("log")
plt.show() 

