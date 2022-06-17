import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import statistics as stat
from scipy import linalg
import math
from sklearn.neural_network import MLPRegressor
import matplotlib
import time

n_rounds = 20
lambda_value = 10
k_value = 20 
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

sample_train = sample[0:n_train,:]

scaler = preprocessing.StandardScaler().fit(np.array(sample_train))
sample = scaler.transform(np.array(sample))

# 
print('MLP')
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
regr = MLPRegressor(hidden_layer_sizes=(T,), activation='relu', solver='lbfgs', max_iter=2000,alpha=lambda_value).fit(sample_train, label_train)
end_time = time.time()
mlp_time = end_time-start_time

print("RVFL")
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
h = np.random.randn(T, p)
hyp_matrix_train = np.matmul(sample_train, np.array(h).transpose())
hyp_matrix_test = np.matmul(sample_test, np.array(h).transpose())
hyp_matrix_train = np.maximum(hyp_matrix_train, 0)
hyp_matrix_test = np.maximum(hyp_matrix_test, 0)
alpha = ridge_regression_train(hyp_matrix_train, label_train, lambda_value)
end_time = time.time()
rvfl_time = end_time-start_time

print("starting rhss")
sample_train, sample_test, label_train, label_test = reset(sample, label, n_train)

start_time = time.time()
hyps_train = []
for j in range(k_value):
    # train the model using rvfl
    h = np.random.randn(T, p)
    # X to hidden layer
    hyp_matrix_train = np.matmul(sample_train, h.transpose())
    # Apply RELU
    hyp_matrix_train = np.maximum(hyp_matrix_train, 0)
    # Hidden layer to intermediate output (random)
    coeff = np.random.randn(T)
    label_pred_train = ridge_regression_predict(hyp_matrix_train, coeff)
    hyps_train.append(label_pred_train)
hyps_train = np.array(hyps_train).transpose()
# Find alpha
alpha = ridge_regression_train(hyps_train, label_train, 0)
end_time = time.time()
rhss_time = end_time-start_time

print('Showing Time')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['MLP', 'RVFL', 'RHSS-KRR']
times = [mlp_time,rvfl_time,rhss_time]
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
