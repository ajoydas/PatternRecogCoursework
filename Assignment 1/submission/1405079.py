#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:16:52 2018

@author: ajoy
"""

import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from sklearn.metrics import accuracy_score


df_train = pd.read_csv('dataset/Train.txt', skiprows = [0], header=None, delimiter = '\s*')
df_test = pd.read_csv('dataset/Test.txt', header=None, delimiter = '\s*')

train = df_train.values
test = df_test.values 

binary_train = df_train[df_train[df_train.keys()[-1]] != 3].values
binary_test = df_test[df_test[df_test.keys()[-1]] != 3].values
col = np.ones(binary_train.shape[0]).reshape(binary_train.shape[0],1)
binary_train = np.hstack((binary_train[:,:3], col, binary_train[:,3:]))

col = np.ones(binary_test.shape[0]).reshape(binary_test.shape[0],1)
binary_test = np.hstack((binary_test[:,:3], col, binary_test[:,3:]))

#w = [0.0 for i in range(len(binary_train[0])-2)]
#w.append(1.0)

w = np.random.random_sample((len(binary_train[0])-1))

weights = w.copy()

def predict(row, weights):
	activation = 0
	for i in range(len(weights)-1):
		activation += weights[i] * row[i]
	return 1 if activation >= 0.0 else 2

def add_to_weights(weights, row):
    for i in range(len(weights)-1):
        weights[i] += row[i]
    return weights

def sub_from_weights(weights, row):
    for i in range(len(weights)-1):
        weights[i] -= row[i]
    return weights

def test_accuracy(binary_test, weights):
    error = 0
    for row in binary_test:
        predicted = predict(row, weights)
    #    print("Class : "+ str(row[-1]) +" Predicted: "+ str(predicted))
        if predicted != row[-1]:
            error += 1
        
    accuracy = 1 - error/binary_train.shape[0]
    print(accuracy)
    return accuracy


def basic_perceptron(weights, binary_train, binary_test):  
    while True :
        print(weights)
        new_weights = weights.copy()
        for row in binary_train:
            predicted = predict(row, weights)
    #        print("Class : "+ str(row[-1]) +" Predicted: "+ str(predicted))
            if predicted != row[-1]:
                if predicted == 1:
                    new_weights = sub_from_weights(new_weights, row)
                else:
                    new_weights = add_to_weights(new_weights, row)
        print(new_weights)
        if (weights == new_weights).all():
            break
        weights = new_weights
    
    test_accuracy(binary_test, weights)

def rp_perceptron(weights, binary_train, binary_test):  
    count = 0
    while True :
    #    print(weights)
        for row in binary_train:
            predicted = predict(row, weights)
    #        print("Class : "+ str(row[-1]) +" Predicted: "+ str(predicted))
            if predicted != row[-1]:
                count = 0
                if predicted == 1:
                    weights = sub_from_weights(weights, row)
                else:
                    weights = add_to_weights(weights, row)
                
                print(weights)
    
            if count == binary_train.shape[0]:
                break
            
            count += 1
        
        if count == binary_train.shape[0]:
            break
    
    test_accuracy(binary_test, weights)


def pocket_perceptron(weights, binary_train, binary_test):  
    best_weights =  weights
    best_accuracy = 0
    count = 0
    K = 100
    k = 0
    while True:
        #    print(weights)
        for row in binary_train:
            predicted = predict(row, weights)
    #        print("Class : "+ str(row[-1]) +" Predicted: "+ str(predicted))
            if predicted != row[-1]:
                k += 1
                count = 0
                if predicted == 1:
                    weights = sub_from_weights(weights, row)
                else:
                    weights = add_to_weights(weights, row)
                
                print(weights)
                
                accuracy = test_accuracy(binary_train, weights)
                if accuracy > best_accuracy:
                    best_weights, best_accuracy = weights, accuracy 
                    
                print(accuracy)
    
            if k == K or count == binary_train.shape[0]:
                break
            
            count += 1
            
        if k == K or count == binary_train.shape[0]:
            break
    
    weights = best_weights
    
    test_accuracy(binary_test, weights)


weights = w.copy()
basic_perceptron(weights, binary_train, binary_test)
rp_perceptron(weights, binary_train, binary_test)
pocket_perceptron(weights, binary_train, binary_test)



col = np.ones(train.shape[0]).reshape(train.shape[0],1)
train = np.hstack((train[:,:3], col, train[:,3:]))

col = np.ones(test.shape[0]).reshape(test.shape[0],1)
test = np.hstack((test[:,:3], col, test[:,3:]))

def kesler_perceptron(train, test):
    count=0
    attrs = train.shape[1]-2
    Y = np.unique(train[:,-1])
    clvalues = len(Y)
    samples = np.empty((0,(attrs+1)*clvalues))
    
    for i in train[:, :train.shape[1]-1]:
        clvalue = int(train[count, -1])
        
        new_sample=np.zeros(((attrs+1)*clvalues))
        new_sample[(clvalue-1)*(attrs+1) : clvalue*(attrs+1)]=i
    
        for j in range(clvalues):
            if((j+1) != clvalue):
                x=new_sample.copy()
                x[j*(attrs+1) : (j+1)*(attrs+1)] = -i
                samples = np.vstack([samples, x])
    
        count+=1
    
    
    count=0
    learning_rate= 0.5
    weights =np.random.random_sample(((attrs+1)*clvalues,))
    while(True):
        for i in samples:
            predicted =np.dot(i,weights)
            
            if predicted < 0:
                weights=weights+(i*learning_rate)
                count = 0
            
            if(count == samples.shape[0]):
                break
            
            count+=1
        if(count == samples.shape[0]):
                break
    
    
    predicted=list()
    for row in test[:, :test.shape[1]-1]:
        max_val = 0
        best_prediction = 0
        for k in range(clvalues):
            new_sample=np.zeros(((attrs+1)*clvalues))
            new_sample[k*(attrs+1) : (k+1)*(attrs+1)] = row
            prediction_value = np.dot(new_sample,weights)
            if prediction_value > max_val:
                max_val = prediction_value
                best_prediction = k+1
        
        predicted.append(best_prediction)
        
    
#    print(predicted)
    print(accuracy_score(test[:, -1], predicted))

kesler_perceptron(train, test)
    
    