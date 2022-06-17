import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import statistics as stat
from scipy import linalg
import math
from sklearn.neural_network import MLPRegressor
import matplotlib

n_rounds = 20
lambda_value = 10
k_values = [3,6,12,24,48,60,80,100]
T = 20

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

print("check")
# Baseline Network (not RVFL) 
error_test = []
sum_test = 0
for i in range(n_rounds):
    # reset variables
    sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
    
    regr = MLPRegressor(hidden_layer_sizes=(T,), activation='relu', solver='lbfgs', max_iter=2000,alpha=lambda_value).fit(sample_train, label_train)
    label_pred_test = regr.predict(sample_test)
    mse_test = find_error(label_pred_test,label_test)
    # store error in list
    error_test.append(mse_test)
    # compute sum of errors
    sum_test += mse_test
err_base = [sum_test/n_rounds for _ in range(len(k_values))]
std_base = [stat.stdev(error_test) for _ in range(len(k_values))]

print("starting RVFL")
# RVFL: fix hidden layer size to T
k_value = T
avg_test_err_rvfl = []
stdev_test_rvfl = []

sum_test = 0
error_test = []
for i in range(n_rounds):
    # reset variables
    sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
    # train the model using rvfl
    h = np.random.randn(T, p)
    hyp_matrix_train = np.matmul(sample_train, np.array(h).transpose())
    hyp_matrix_test = np.matmul(sample_test, np.array(h).transpose())
    # Apply RELU
    hyp_matrix_train = np.maximum(hyp_matrix_train, 0)
    hyp_matrix_test = np.maximum(hyp_matrix_test, 0)
    # Find alpha
    alpha = ridge_regression_train(hyp_matrix_train, label_train, lambda_value)
    label_pred_test = ridge_regression_predict(hyp_matrix_test, alpha)
    mse_test = find_error(label_pred_test,label_test)
    # store error in list
    error_test.append(mse_test)
    # compute sum of errors
    sum_test += mse_test
avg_test_err_rvfl.append(sum_test/n_rounds)
stdev_test_rvfl.append(stat.stdev(error_test))

# print("check RP")
# # Baseline Network (not RVFL)
# err_rp = []
# std_rp = []
# for k_value in k_values:
#     proj = np.random.randn(k_value, p)
#     proj_sample = np.matmul(sample, proj.transpose())
#     error_test = []
#     sum_test = 0
#     for i in range(n_rounds):
#         # reset variables
#         sample_train, sample_test, label_train, label_test = reset(proj_sample, label, n_train)
#         regr = MLPRegressor(hidden_layer_sizes=(T,), activation='relu', solver='lbfgs', max_iter=2000,alpha=lambda_value).fit(sample_train, label_train)
#         label_pred_test = regr.predict(sample_test)
#         mse_test = find_error(label_pred_test,label_test)
#         # store error in list
#         error_test.append(mse_test)
#         # compute sum of errors
#         sum_test += mse_test
#     err_rp.append(sum_test/n_rounds)
#     std_rp.append(stat.stdev(error_test))
# err_rp = np.array(err_rp)
# std_rp = np.array(std_rp)
    
print("starting RVFL+RHS")
# RVFL+RHS
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
            # train the model using rvfl
            h = np.random.randn(T, p)
            # X to hidden layer
            hyp_matrix_train = np.matmul(sample_train, h.transpose())
            hyp_matrix_test = np.matmul(sample_test, h.transpose())
            # Apply RELU
            hyp_matrix_train = np.maximum(hyp_matrix_train, 0)
            hyp_matrix_test = np.maximum(hyp_matrix_test, 0)
            # Hidden layer to intermediate output (random)
            coeff = np.random.randn(T)
            label_pred_test = ridge_regression_predict(hyp_matrix_test, coeff)
            hyps_test.append(label_pred_test)
            label_pred_train = ridge_regression_predict(hyp_matrix_train, coeff)
            hyps_train.append(label_pred_train)
            matrix_tests.append(hyp_matrix_test)
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
    


plt.xlabel('Value of k',fontsize=18)
plt.ylabel('Testing RMSE',fontsize=18)
plt.plot(k_values, err_base,linestyle='--',color='g', label="MLP")
# plt.plot(k_values, err_rp,linestyle='--',color='r', label="RP-MLP")
# plt.fill_between(k_values, err_rp-std_rp,err_rp+std_rp,color='r',alpha=0.2)
plt.plot(k_values, avg_test_err_rvfl[0]*np.ones(len(k_values)),linestyle='--',color='C0', label="RVFL")
plt.plot(k_values, err_rvfl_rhs,color='darkorange', label="RHSS-MLP")
plt.fill_between(k_values, np.array(err_base)-np.array(std_base),np.array(err_base)+np.array(std_base),color='g',alpha=0.2)
plt.fill_between(k_values, avg_test_err_rvfl[0]*np.ones(len(k_values))-stdev_test_rvfl*np.ones(len(k_values)),avg_test_err_rvfl[0]*np.ones(len(k_values))+stdev_test_rvfl*np.ones(len(k_values)),color='C0', alpha=0.2)
plt.fill_between(k_values, np.array(err_rvfl_rhs)-np.array(std_rvfl_rhs),np.array(err_rvfl_rhs)+np.array(std_rvfl_rhs),color='darkorange', alpha=0.2)
plt.legend(loc="upper right")
matplotlib.rc('font', size=18)
plt.xscale("log")
plt.show() 
