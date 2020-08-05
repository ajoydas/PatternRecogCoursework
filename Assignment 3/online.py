import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.stats import multivariate_normal

trainfile = 'Evaluation/train.txt' 
testfile = 'Evaluation/test.txt' 
outfile = 'Evaluation/out.txt'
paramfile = 'param.txt'

def readfile(filename):
    file = open(filename, 'rb')
    
    byte = [int(file.read(1)), int(file.read(1)), int(file.read(1))]
    windows = [byte.copy()]
    while 1:
        byte.pop(0)    
        byte_s = file.read(1)
        if not byte_s:
             break
         
        byte.append(int(byte_s))
        windows.append(byte.copy())
    return windows

class Channel:
    def __init__(self, h1, h2, mu, sigma):
        self.h1 = h1
        self.h2 = h2
        self.mu = mu
        self.sigma = sigma
        
    def to_str(self, window):
        return window[0]+window[1]+window[2]

    def add_noise(self, xk, xk_1):
        nk = np.random.normal(self.mu, self.sigma, 1)[0]
        xk = self.h1 * xk + self.h2 * xk_1 + nk
        return xk
    
    def apply_channel(self, windows):
        mapping = [[[0,0],[0,0]],[[0,0],[0,0]]]
  
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mapping[i][j][k] = idx
                    print(str(i)+str(j)+str(k)+": "+str(idx))
                    idx +=1
                    
        X = list()
        for i in range(8):
            X.append(list())
        
        for window in windows:
            xk = self.add_noise(window[2], window[1])
            xk_1 = self.add_noise(window[1], window[0])
            
            mapped = mapping[window[2]][window[1]][window[0]]
            X[mapped].append(np.array([xk,xk_1]))
        return X
    
    def apply_channel_test(self, windows):
        X = list()
        for window in windows:
            xk = self.add_noise(window[2], window[1])
            xk_1 = self.add_noise(window[1], window[0])
            X.append(np.array([xk,xk_1]))
        return X

######## Reading Channel Param #######
file = open(paramfile, 'r')
h1 = float(file.readline())
h2 = float(file.readline())
mu = float(file.readline())
sigma = float(file.readline())

######## Applying Channel #######
windows = readfile(trainfile)
windows_test = readfile(testfile)

channel = Channel(h1, h2, mu, sigma)
X = channel.apply_channel(windows)

X_test = channel.apply_channel_test(windows_test)
X_test = np.array(X_test)

m = 8
N = X_test.shape[0]

######## Calculate means, covs, priors #######
means = []
covs = []
prior = []

for i in range(m):
    X[i] = np.array(X[i]).T

for i in range(m):
    means.append(np.array([X[i][0].mean(),X[i][1].mean()]))  
    covs.append(np.cov(X[i]))
    prior.append(X[i].shape[1])     

prior = np.array(prior)
prior = prior / prior.sum()

####### Applying Viterbi #######
D = np.zeros((m, N))
for k in range(m):    
    p_xi_wk = multivariate_normal.pdf(X_test[0], means[k], covs[k])
    D[k][0] = np.log10(prior[k] * p_xi_wk +eps)


mapping = ['000','001','010','011', '100','101','110','111']
parent = [[0,1],[2,3],[4,5],[6,7],[0,1],[2,3],[4,5],[6,7]]

for i in range(1,N):
    for k in range(m):
        p_xi_wk = multivariate_normal.pdf(X_test[i], means[k], covs[k])
        
        prev_path =  max(D[parent[k][0]][i-1], D[parent[k][1]][i-1])
        D[k][i] = prev_path + np.log10(prior[k] * p_xi_wk +eps)

####### Finding Max Path #######
#out = []
#for i in range(0,N):
#    out.append(mapping[D[:,i].argmax()])
        
out = [D[:,N-1].argmax()]
k = 0
for i in range(N-2,-1,-1):
    prev = out[k]
    if D[parent[prev][0]][i] > D[parent[prev][1]][i]:
        maxarg = parent[prev][0]
    else:
        maxarg = parent[prev][1]
    
    out.append(maxarg)
    k +=1
        
out.reverse()
for i in range(0,N):
    out[i] = mapping[out[i]]
    
out_byte = out[0][::-1]
for i in range(1,N):
    out_byte += out[i][0] 

######## Matching #######
file = open(testfile, 'r')
line = str(file.readline())

if out_byte == line:
    print("Matched Fully..")
else:
    print("Didn't Matched Fully..")

count = 0    
for i in range(len(line)):
    if out_byte[i] == line[i]:
        count += 1
print(count/len(line))

######## Writing to output file #######
file = open(outfile, 'w')
file.write(out_byte)
file.flush()
file.close()
