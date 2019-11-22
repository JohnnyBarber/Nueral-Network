# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:28:13 2019

@author: Jiaqi Li
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

def sigmoid(x):
    s = 1/(1+np.exp(x))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

def propagate (w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    
    cost = np.squeeze(cost)
    grads = {'dw': dw, 'db': db}
    
    return grads, cost

def optimize(w,b,X,Y,num_iter,alpha):
    costs = []
    
    for i in range(num_iter):
        grads, costs = propagate(w,b,X,Y)
        dw = grads['dw']
        db = grads['db']
        w -= alpha*dw
        b -= alpha*db
        if i%100 == 0:
            costs.append(costs)
    
   params = {'w':w, 'b':b}
   grads = {'dw':dw, 'db':db}
   return params, grads, costs

def predict(w,b,X):
    m = X.shape[1]
    Y_predict = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        Y_predict[0,i] = 1 if A[0,i] > 0.5 else 0
    
    return Y_predict

def logit_model(X_train,Y_train,X_test,Y_test,num_iter = 20000,alpha = 0.5):
    dim = X_train.reshape(X_train.shape[0],-1).T.shape[1]
    w,b = initialize_with_zeros(dim)
    
    params, grads, costs = optimize(w,b,X_train,Y_train,num_iter,alpha)
    
    w = params['w']
    b = params['b']
    
    Y_predict_train = predict(w,b,X_train)
    Y_predict_test = predict(w,b,X_test)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {'costs':costs,
         'Y_predict_test': Y_predict_test,
         'Y_predict_train': Y_predict_train,
         'w': w,
         'b': b,
         'learning rate': alpha,
         'num_iteration': num_iter}
    return d
    
