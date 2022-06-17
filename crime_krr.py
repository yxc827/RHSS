import numpy as np
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import statistics as stat
import matplotlib
from scipy import linalg
import math
import sklearn
from scipy.stats import multivariate_normal


n_rounds = 20
lambda_value = 0.1
k_values = [2,5,8,10,20,30,70,100]
gamma_value = 0.001

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

def kernel(X_train, X_test):
    M = sklearn.metrics.pairwise.rbf_kernel(X_train,gamma=gamma_value)
    N = sklearn.metrics.pairwise.rbf_kernel(X_test,X_train,gamma=gamma_value)
    return M, N

def reset(sample, label, n_train):
    sample_train = sample[0:n_train,:]
    sample_test = sample[n_train:,:]
    label_train = label[0:n_train]
    label_test = label[n_train:]  
    return sample_train, sample_test, label_train, label_test

class GaussianRFF():
    def __init__(self, D, d=0, gamma=0.001, mu=0, sigma=1, **args):
        self.d = d # placeholder for feature dimension d
        self.gamma = gamma
        self.D = D # dimension D of randomized feature map z(x)
        self.mu = mu
        self.sigma = sigma
        self.alpha = np.ones(d)
        self.ZX = np.diag(np.ones(d)) # placeholder for z(x) with x being training data
        self.W = np.diag(np.ones(d)) # placeholder for W
        self.B = np.ones(self.D) # placeholder for bias term
        
    def ridge_regression_train(X,y,lambda_value):
        X = np.array(X)
        XTy = np.matmul(X.transpose(),y)
        XTXl = np.matmul(X.transpose(),X)+lambda_value*np.identity(X.shape[1])
        alpha = np.matmul(linalg.pinv(XTXl), XTy)
        return alpha
    
    def ridge_regression_predict(X,alpha):
        return np.matmul(X,alpha)    
        
    def fit(self, X, y, **args):
        _, self.d = X.shape
        self.mu = np.zeros(self.d)
        self.sigma = (self.gamma/(2*math.pi**2))*np.diag(np.ones(self.d))
        
        rv = multivariate_normal(mean = self.mu, cov = self.sigma)
        W = rv.rvs(self.D)
        B = np.random.uniform(0, 2*math.pi, self.D)
        self.W = W
        self.B = B
        
        WX = np.matmul(W, X.transpose())
        WXpB = WX.transpose() + B
        ZX = math.sqrt(2/self.D)*np.cos(WXpB)
        approxK = ZX
        
        self.ZX = ZX
        self.alpha = ridge_regression_train(approxK, y, 0.01)
        return self

    def predict(self, X, **args):
        WXtt = np.matmul(self.W, X.transpose())
        WXttpB = WXtt.transpose() + self.B
        ZXtt = math.sqrt(2/self.D)*np.cos(WXttpB)
        approxKtt = ZXtt.transpose()
        return ridge_regression_predict(approxKtt.transpose(),self.alpha)
        
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

# Kernel Ridge Regression baseline
er = []
for i in range(n_rounds):
    # reset variables
    sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
    
    # kernel ridge regression
    reg = KernelRidge(kernel='rbf',gamma=gamma_value,alpha=lambda_value).fit(sample_train, label_train) 
    pred_label = reg.predict(sample_test)
    error_kernelridge_test = find_error(pred_label, label_test)
    er.append(error_kernelridge_test)
    
stdev_kernelridge_test = stat.stdev(er)
error_kernelridge_test = sum(er)/len(er)
avg_kernelridge_err = np.array([error_kernelridge_test for _ in range(len(k_values))])
stdev_kernelridge = np.array([stdev_kernelridge_test for _ in range(len(k_values))])

print("starting rff")
# RFF sklearn + ridge regression
avg_rffsk_err = []
stdev_rffsk = []
for k_value in k_values:
    sum_test = 0
    error_test = []
    for i in range(n_rounds):
        print(i)
        # reset variables
        sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
        rff = GaussianRFF(gamma=gamma_value,D=k_value, normalize=True).fit(sample_train, label_train)
        pred_label = rff.predict(sample_test)
        
        error_ridge_test = find_error(pred_label, label_test)
        error_test.append(error_ridge_test)
    avg_rffsk_err.append(sum(error_test)/n_rounds)
    stdev_rffsk.append(stat.stdev(error_test))
avg_rffsk_err = np.array(avg_rffsk_err)
stdev_rffsk = np.array(stdev_rffsk)

# print("check RP")
# err_rp = []
# std_rp = []
# for k_value in k_values:
#     proj = np.random.randn(k_value, p)*0.1
#     proj_sample = np.matmul(sample, proj.transpose())
#     error_test = []
#     sum_test = 0
#     for i in range(n_rounds):
#         # reset variables
#         sample_train, sample_test, label_train, label_test = reset(proj_sample, label, n_train)
#         regr = KernelRidge(kernel='rbf',gamma=gamma_value,alpha=lambda_value).fit(sample_train, label_train)
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

print("starting rhsk")
# RHSK
avg_test_err_rhsk = []
stdev_test_rhsk = []
mu, sigma = 0, 1
lambda_value = 0
for k_value in k_values:
    p = n_train
    sum_test = 0
    error_test = []
    for i in range(n_rounds):
        print(i)
        # reset variables
        sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
        sample_train, sample_test = kernel(sample_train, sample_test)
        # sample_train = sample_train[:,:m]
        # sample_test = sample_test[:,:m]
        # # train the model using randomized alg
        h = sigma*np.random.randn(n_train, k_value) + mu
        hyp_matrix_train = np.matmul(sample_train, np.array(h))
        hyp_matrix_test = np.matmul(sample_test, np.array(h))
        # Find alpha
        alpha = ridge_regression_train(hyp_matrix_train, label_train, 0)
        #print(alpha)
        label_pred_test = ridge_regression_predict(hyp_matrix_test, alpha)
        mse_test = find_error(label_pred_test,label_test)
        # store error in list
        error_test.append(mse_test)
        # compute sum of errors
        sum_test += mse_test
    avg_test_err_rhsk.append(sum_test/n_rounds)
    stdev_test_rhsk.append(stat.stdev(error_test))
avg_test_err_rhsk = np.array(avg_test_err_rhsk)
stdev_test_rhsk = np.array(stdev_test_rhsk)  

plt.xlabel('Value of k',fontsize=18)
plt.ylabel('Testing RMSE',fontsize=18)
plt.plot(k_values, avg_kernelridge_err,linestyle='--',color='g', label="KRR")
# plt.plot(k_values, err_rp,linestyle='--',color='r', label="RP-KRR")
plt.plot(k_values, avg_rffsk_err,linestyle='--',color='C0', label="RFF-KRR")
plt.plot(k_values, avg_test_err_rhsk,color='darkorange', label="RHSS-KRR")
# plt.fill_between(k_values, err_rp-std_rp,err_rp+std_rp,color='r',alpha=0.2)
plt.fill_between(k_values, avg_test_err_rhsk-stdev_test_rhsk,avg_test_err_rhsk+stdev_test_rhsk,color='darkorange', alpha=0.2)
plt.fill_between(k_values, avg_kernelridge_err-stdev_kernelridge,avg_kernelridge_err+stdev_kernelridge,color='g', alpha=0.2)
plt.fill_between(k_values, avg_rffsk_err-stdev_rffsk,avg_rffsk_err+stdev_rffsk,color='C0', alpha=0.2)
matplotlib.rc('font', size=18)
plt.legend(loc="upper right")
plt.xscale("log")
plt.show() 

