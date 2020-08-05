import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import time

a = 1
mu = 0.05

df_train = pd.read_csv('data/trainNN.txt', header=None, delimiter = '\s*')
df_test = pd.read_csv('data/testNN.txt', header=None, delimiter = '\s*')

#df_train = pd.read_csv('evaluation/trainNN.txt', header=None, delimiter = '\s*')
#df_test = pd.read_csv('evaluation/testNN.txt', header=None, delimiter = '\s*')

D = df_train.shape[1]-1

df_train_y = pd.get_dummies(df_train[D])
df_train_x = df_train.drop(D,axis = 1)


train_x = df_train_x.values
train_y = df_train_y.values

min_max_scaler = preprocessing.StandardScaler()
train_x = min_max_scaler.fit_transform(train_x)

col = np.ones(train_x.shape[0]).reshape(train_x.shape[0],1)
train_x = np.hstack((train_x, col))


df_test_y = pd.get_dummies(df_test[D])
df_test_x = df_test.drop(D,axis = 1)

test_x = df_test_x.values
test_y = df_test_y.values

test_x = min_max_scaler.transform(test_x)
col = np.ones(test_x.shape[0]).reshape(test_x.shape[0],1)
test_x = np.hstack((test_x, col))
#test = df_test.values 



def sigmoid(x):
    return 1 / (1 + np.exp(-a*x))

def sigmoid_der(x):
    return a*sigmoid(x)*(1-sigmoid(x))

layer = [train_x.shape[1], 3, 4, 5, train_y.shape[1]]

weight = []
for i in range(len(layer)-1):
    w = np.random.uniform(-1,1,(layer[i],layer[i+1]))
    weight.append(w)

maxItr = 500
y = []
v = []
delta = []

min_err = np.inf
best_w = []

start = time.time()
for t in range(maxItr):
    delw = []
    errs = 0
    for r in range(len(layer)-1):
        delw.append([0]*layer[r+1])

    for i in range(train_x.shape[0]):   
        
        input = [train_x[i]]
        y = []
        v = []
        for r in range(len(layer)-1):
            w = weight[r]
            out = np.matmul(input[r], w)
            v.append(out)
            out = sigmoid(out)        
            
            y.append(out)
            input.append(out)
               
        l = len(layer)-1
        
        errs += (0.5 * (y[l-1] - train_y[i])*(y[l-1] - train_y[i])).sum()
        
        delta = []
        d = []
        for j in range(layer[-1]):
            ej = y[l-1][j] - train_y[i][j]
            d.append(ej*sigmoid_der(v[l-1][j]))
            
        delta.append(d)
        for r in range(len(layer)-2, 0, -1):
            d = []
            l = layer[r]
            for j in range(l):
#                print(str(r)+" "+str(j))
                
                dot = np.dot(delta[len(layer)-r-2], weight[r][j])
                d.append(dot*sigmoid_der(v[r-1][j]))
                
            delta.append(d)
            
        delta.reverse()
        
        for r in range(len(layer)-1):
            w = []
            l = layer[r+1]
            for j in range(l):
#                print(str(r)+" "+str(j))
                w = delta[r][j]*input[r]
                delw[r][j] += w
                
    
    for r in range(len(layer)-1):
        l = layer[r+1]
        for j in range(l):
#            print(str(r)+" "+str(j))
            weight[r][:,j] = weight[r][:,j] - mu * delw[r][j] 
    
    if errs < min_err:
            min_err = errs
            best_w = weight
    
    if t % 100 == 0:
        print("Error ", errs, " Time: ", time.time() - start)
        
print("Training finished, time needed: ", time.time() - start)

weight = best_w

def result(data, data_x, weight):
    output = []
    for i in range(data_x.shape[0]):   
        input = [data_x[i]]
        for r in range(len(layer)-1):
            w = weight[r]
            out = np.matmul(input[r], w)
            out = sigmoid(out)        
            input.append(out)
        
    #    print(input[len(layer)-1])
        output.append(input[len(layer)-1].argmax()+1)
    
    score = accuracy_score(data.iloc[:,-1], output)
    print("Accuracy:"+str(score))
    
    print("Miss classified samples:")
    for i in range(len(output)):
        if data.iloc[i,-1] != output[i]:
            print(data.iloc[i,:])
            print("Predicted: "+str(output[i]))


result(df_train, train_x, best_w)
result(df_test, test_x, best_w)
















        
        
        

