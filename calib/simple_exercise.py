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
from matplotlib import pyplot as plt

def test_simple():
    
    num_sample=1000
    
    x = np.expand_dims(np.linspace(-2*np.pi, 2*np.pi, num=2000), axis=1)
    np.random.RandomState(0).shuffle(x)
    order=np.argsort(x.flatten())
    y=x+5*np.sin(x)
    
    
    model=nn.Sequent(nn.Linear(1,150),nn.ReLU(),nn.Linear(150,1))
    
    opt=optim.Adam(model.parameters(),lr=0.005)
    
    loss_func=nn.MseLoss()
    def run(model):
        X=Tensor(x)
        out=model(X)
        return out.to_numpy()
    
    
    def train(batch_size=200,epochs=10000):
        for i in range(0,epochs):
            l=(i*batch_size)%num_sample
            y1=y[l:l+batch_size]
            x1=x[l:l+batch_size]
            real=Tensor(y1)
            X=Tensor(x1)
            pred=model(X)
            loss=loss_func(pred,real)
            if i%20==0:
                print(loss.data.cachedData)
            if i%100==0:
                out_put=run(model)
                plt.plot(x[order], y[order],color="blue")
                plt.plot(x[order], out_put[order],color="red")
                plt.show()
            loss.backward()
            opt.step()
    
    train(200,10000)
    
test_simple()