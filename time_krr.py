import numpy as np
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from scipy import linalg
import math
import sklearn
from scipy.stats import multivariate_normal
import time

lambda_value = 0.1
k_value = 20
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

def kernel_v2(X):
    M = sklearn.metrics.pairwise.rbf_kernel(X,gamma=gamma_value)
    return M

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

# 
print('KRR')
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
reg = KernelRidge(kernel='rbf',gamma=gamma_value,alpha=lambda_value).fit(sample_train, label_train) 
end_time = time.time()
krr_time = end_time-start_time

pred_label = reg.predict(sample_test)
error_kernelridge_test = find_error(pred_label, label_test)


print("starting rff")
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
rff = GaussianRFF(gamma=gamma_value,D=k_value, normalize=True).fit(sample_train, label_train)
end_time = time.time()
rff_time = end_time-start_time

pred_label = rff.predict(sample_test)
error_ridge_test = find_error(pred_label, label_test)

print("check RP")
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
proj = np.random.randn(k_value, p)
proj_sample = np.matmul(sample, proj.transpose())
sample_train, sample_test, label_train, label_test = reset(proj_sample, label, n_train)
regr = KernelRidge(kernel='rbf',gamma=gamma_value,alpha=lambda_value).fit(sample_train, label_train)
end_time = time.time()
rp_time = end_time-start_time

label_pred_test = regr.predict(sample_test)
mse_test = find_error(label_pred_test,label_test)


print("starting rhsk")
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

mu, sigma = 0, 1

start_time = time.time()
h = sigma*np.random.randn(n_train, k_value) + mu
sample_train = kernel_v2(sample_train)
hyp_matrix_train = np.matmul(sample_train, np.array(h))
alpha = ridge_regression_train(hyp_matrix_train, label_train, 0)
end_time = time.time()
rhsk_time = end_time-start_time

sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)
sample_train, sample_test = kernel(sample_train, sample_test)
hyp_matrix_test = np.matmul(sample_test, np.array(h))
label_pred_test = ridge_regression_predict(hyp_matrix_test, alpha)
mse_test = find_error(label_pred_test,label_test)

print('Showing Time')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['KRR', 'RP-KRR', 'RFF-KRR', 'RHSS-KRR']
times = [krr_time,rp_time,rff_time,rhsk_time]
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
