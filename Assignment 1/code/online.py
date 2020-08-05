#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:16:52 2018

@author: ajoy
"""

import pandas as pd
import numpy as np
eps = np.finfo(float).eps

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

w = [0.0 for i in range(len(binary_train[0])-2)]
w.append(1.0)

weights = w.copy()
# Make a prediction with weights
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
    if weights == new_weights:
        break
    weights = new_weights

test_accuracy(binary_test, weights)


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

weights = w.copy()

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


test_accuracy(binary_test, weights)

col = np.ones(train.shape[0]).reshape(train.shape[0],1)
train = np.hstack((train[:,:3], col, train[:,3:]))


count=0
att = train.shape[1]-2
Y = np.unique(train[:,-1])
clss = len(Y)
finalMat = np.empty((0,(att+1)*clss))

for i in train[:, :train.shape[1]-1]:
#    print(i)
    a=np.zeros(((att+1)*clss))
    #print(Y[count])
    #print(count)
    classVal = int(train[count, -1])
    #print(classVal)
    a[(classVal-1)*(att+1) : classVal*(att+1)]=i
    #print([a])
    for j in range(clss):
        if( (j+1) != classVal):
            x=a.copy()
            x[j*(att+1) : (j+1)*(att+1)] = -i
            finalMat = np.vstack([finalMat, x])
            #print(x)
    count+=1


counter=0
maxCounter=finalMat.shape[0]
constantTerm= 0.5
w =np.random.random_sample(((att+1)*clss,))
while(True):
    for i in finalMat:
        product=np.dot(i,w)
        counter+=1
        if(product < 0):
            w=w+(i*constantTerm)
            #print(w)
            counter = 0
        
        if(counter == maxCounter):
            break
    
    
    if(counter == maxCounter):
            break


predictedOutput=[]
for eachrow in test:
    val=0
    classed=0
    for k in range(clss):
        a=np.zeros(((att+1)*clss))
        a[k*(att+1) : (k+1)*(att+1)] = eachrow
        got=np.dot(a,w)
        if(val<got):
            val=got
            classed=k+1
    predictedOutput.append(classed)
    #print(classed)
    

print(predictedOutput)

from sklearn.metrics import classification_report

target_names = ['class 1', 'class 2', 'class 3']

print(classification_report(train[:, -1], predictedOutput, target_names=target_names))

from sklearn.metrics import accuracy_score
print(accuracy_score(train[:, -1], predictedOutput))
    
    