import numpy as np
classes = []
with open('data/trainNN.txt') as file:
    train_data = []
    for i in file:
        data = [float(j) for j in i.split()]
        train_data.append(data)
        if (int(data[-1]) not in classes):
            classes.append(data[-1])
            
numclasses = len(classes)
numfeatures = len(train_data[0]) - 1
numdata = len(train_data)

print(numclasses, numfeatures, numdata)
#print(train_data[0])

train_features = []
train_labels = []

for i in range(numdata):
    f = [train_data[i][j] for j in range(numfeatures)]
    f.append(1.00)
    train_features.append(f)
    l = list(np.zeros(numclasses))
    l[int(train_data[i][-1]) - 1] = 1
    train_labels.append(l)
    
train_features = np.array(train_features)
train_labels = np.array(train_labels)

#print(train_labels[:,0])


with open('data/testNN.txt') as file:
    test_data = []
    for i in file:
        data = [float(j) for j in i.split()]
        test_data.append(data)
            
test_features = []
test_labels = []

for i in range(len(test_data)):
    f = [test_data[i][j] for j in range(numfeatures)]
    f.append(1.00)
    test_features.append(f)
    l = list(np.zeros(numclasses))
    l[int(test_data[i][-1]) - 1] = 1
    test_labels.append(l)
    
test_features = np.array(test_features)
test_labels = np.array(test_labels)

print(len(test_features[0]))

for i in range(numfeatures):
    train_features[:, i] = (train_features[:, i] - np.mean(train_features[:, i])) / np.std(train_features[:, i])
    test_features[:, i] = (test_features[:, i] - np.mean(test_features[:, i])) / np.std(test_features[:, i])
    
print(train_features)

from numpy import exp
layout = [len(train_features[0]), 8, 8, 8, len(train_labels[0])]


weights = []
delta = []
dw = []
activations = []
lr = 0.01


for i in range(len(layout)-1):
    w = 2 * np.random.rand(layout[i], layout[i + 1]) - 1
    weights.append(np.array(w))
    
for i in range(len(weights)):
    print(weights[i].shape)


def sigmoid(x):
    return 1 / (1 + exp(-x))

def derivative(x):
    return x * (1 - x)

def forward_propagation(ip, delta, dw, activations):
    op = ip
    activations.append(op)
    for i in range(len(weights)):
        op = np.dot(op, weights[i])
        op = sigmoid(np.array(op))
        delta.append(op)
        dw.append(op)
        activations.append(op)
    return activations[-1], delta, dw, activations

def backward_propagation(delta_last, delta, dw, activations):
    m = len(delta_last)
    delta[-1] = delta_last.copy()
    
    dw[-1] = np.dot(activations[-2].T, delta[-1]) / m
    
    for i in range(len(weights)-2,-1,-1):
        delta[i] = np.multiply((np.dot(delta[i + 1],weights[i + 1].T)), derivative(activations[i + 1]))
        dw[i] = np.dot(activations[i].T, delta[i]) / m
        
    for i in range(len(weights)):
        weights[i] -= lr * dw[i]


import time
import math

def train():
    start = time.time()
    min_error = math.inf
    lr = 0.01
    
    for epoch in range(50000):
        delta = []
        dw = []
        activations = []
        y_hat, delta, dw, activations = forward_propagation(train_features, delta, dw, activations)
        error = 0.5 * (y_hat - train_labels) * (y_hat - train_labels)
        if (error.sum() < min_error):
            min_error = error.sum()
            np.save('optimal_weights.npy', weights)
        if (epoch % 10000 == 0):
            print("Error ", error.sum(), " Time: ", time.time() - start)
            
        delta_last = (y_hat - train_labels) * derivative(activations[-1])
        backward_propagation(delta_last, delta, dw, activations)
    
    print("Training finished, time needed: ", time.time() - start)
    
train()

def test():
    miss = 0
    y_hat, delta, dw, activations = forward_propagation(train_features, [], [], [])
    for i in range(len(y_hat)):
        value = -1
        prediction = -1
        for j in range(len(y_hat[i])):
            if(y_hat[i][j] > value):
                value = y_hat[i][j]
                prediction = j
        if(train_labels[i][prediction] != 1.00):
            miss+=1
            
    print("Training data total: ", len(train_features), "missed = ", miss)
    
    
    miss = 0
    y_hat, delta, dw, activations = forward_propagation(test_features, [], [], [])
    for i in range(len(y_hat)):
        value = -1
        prediction = -1
        for j in range(len(y_hat[i])):
            if(y_hat[i][j] > value):
                value = y_hat[i][j]
                prediction = j
        if(test_labels[i][prediction] != 1.00):
            miss+=1

    print("Test data total: ", len(test_features), "missed = ", miss)
    
test()
