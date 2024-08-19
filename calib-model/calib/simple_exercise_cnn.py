# -*- coding: utf-8 -*-

import sys

import numpy as np
from numbers import Number
from typing import Tuple,List,Optional,Iterator
from operator import add
from functools import reduce  
import operations as opr 
import optim 
import CGF
import nn
import init
from nn import Parameter,Module
from CGF import Tensor
import pickle
import gc

def ConvBN(a,b,k,s):
   
    return nn.Sequent(nn.Conv(a, b, k,stride=s),nn.BatchNorm2d(b),nn.ReLU())
    

def CNN():
    a=nn.Sequent(ConvBN(3, 16, 7, 4),ConvBN(16, 32, 3, 2),nn.Residual(nn.Sequent(ConvBN(32, 32, 3, 1),ConvBN(32, 32, 3, 1))),ConvBN(32, 64, 3, 2),ConvBN(64,128, 3, 2),
                 nn.Residual(nn.Sequent(ConvBN(128, 128, 3, 1),ConvBN(128, 128, 3, 1))),nn.Flatten(),nn.Linear(128, 128),nn.ReLU(),nn.Linear(128, 10))
    return a
    
def unpickle(file):
    with open(file,'rb') as fo:
        dict=pickle.load(fo,encoding='bytes')
    return dict
if __name__=="__main__":
    
   
    
    d=unpickle('./data/cifar-10-batches-py/data_batch_1')
    y=d[b'labels']
    x=d[b'data']
    d=unpickle('./data/cifar-10-batches-py/data_batch_2')
    y.extend(d[b'labels'])
    x=np.concatenate((x,d[b'data']),0)
    d=unpickle('./data/cifar-10-batches-py/data_batch_3')
    y.extend(d[b'labels'])
    x=np.concatenate((x,d[b'data']),0)
    d=unpickle('./data/cifar-10-batches-py/data_batch_4')
    y.extend(d[b'labels'])
    x=np.concatenate((x,d[b'data']),0)
    d=unpickle('./data/cifar-10-batches-py/data_batch_5')
    y.extend(d[b'labels'])
    x=np.concatenate((x,d[b'data']),0)
    x1=np.zeros((x.shape[0],3,32,32))
    for i in range(0,x.shape[0]):
        x1[i][0]=x[i][:1024].reshape((32,32))
        x1[i][1]=x[i][1024:2048].reshape((32,32))
        x1[i][1]=x[i][2048:3072].reshape((32,32))
    
    
    
    d=unpickle('./data/cifar-10-batches-py/test_batch')
    test_y=d[b'labels']
    test_x=d[b'data']
    x2=np.zeros((test_x.shape[0],3,32,32))
    for i in range(0,test_x.shape[0]):
        x2[i][0]=x[i][:1024].reshape((32,32))
        x2[i][1]=x[i][1024:2048].reshape((32,32))
        x2[i][1]=x[i][2048:3072].reshape((32,32))
    
    model=CNN()
    batch_size=100
    epoch=100
    loss_func=nn.SoftmaxLoss()
    opt=optim.Adam(model.parameters(),lr=0.01)
    for i in range(0,epoch):
        for j in range(0,50000//batch_size):
            train_in=Tensor(x1[j*batch_size:(j+1)*batch_size,:,:,:])
            train_label=Tensor(y[j*batch_size:(j+1)*batch_size])
            pred=model(train_in)
            Loss=loss_func(pred,train_label)
            Loss.backward()
            opt.step()
            
            pred2=np.argmax(pred.offgraph().to_numpy(),1)
            y2=train_label.to_numpy()
            sum=0
            
            for k in range(0,len(y2)):
                if pred2[k]!=y2[k]:
                    sum+=1
            print(Loss.offgraph().to_numpy(),sum/batch_size)
            gc.collect()
            
            
        test_in=Tensor(x2)
        test_label=Tensor(test_y)
        pred=model(test_in)
        pred2=np.argmax(pred.offgraph().to_numpy(),1)
        y2=test_label.to_numpy()
        sum=0
        for k in range(0,len(y2)):
            if pred2[0][k]!=y2[k]:
                sum+=1
        print(sum/len(test_y))
        
        
        
        
        
        
        

    