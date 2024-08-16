# -*- coding: utf-8 -*-

import calib.opr as opr
import numpy as np
class Optimizer:
    def __init__(self,param):
        self.param=param
        
    def step(self):
        raise NotImplementedError
        
    def reset_grad(self):
        for i in self.param:
            i.grad=None
class SGD(Optimizer):
    def __init__(self,param,lr=0.01,momentum=0.0,weight_decay=0.0):
        super().__init__(param)
        self.u={}
        self.lr=lr
        self.m=momentum
        self.weight_decay=weight_decay
        
    def step(self):
        for i in range(0,len(self.param)):
            grad=self.param[i].grad.offgraph().cachedData+self.weight_decay*self.param[i].data.offgraph().cachedData
            
            if i in self.u:
                self.u[i]=np.float32(self.m*self.u[i]+(1-self.m)*grad)
            else:
                self.u[i]=np.float32((1-self.m)*grad)
        for i in range(0,len(self.param)):
            
            self.param[i].cachedData=self.param[i].cachedData-self.lr*self.u[i]
            

class Adam(Optimizer):
    def __init__(self,param,lr=0.01,beta1=0.9,beta2=0.999,ep=1e-8,weight_decay=0.0):
        super().__init__(param)
        self.lr=lr
        self.weight_decay=weight_decay
        self.beta1=beta1
        self.beta2=beta2
        self.u={}
        self.v={}
        self.t=0
        self.ep=ep
        
    def step(self):
        self.t=self.t+1
        for i in range(0,len(self.param)):
            grad=self.param[i].grad.data.offgraph().cachedData+self.weight_decay*self.param[i].data.offgraph().cachedData
            
            if i in self.u:
                self.u[i]=np.float32(self.beta1*self.u[i]+(1-self.beta1)*grad)
                self.v[i]=np.float32(self.beta2*self.v[i]+(1-self.beta2)*grad*grad)
            else:
                self.u[i]=np.float32((1-self.beta1)*grad)
                self.v[i]=np.float32((1-self.beta2)*grad*grad)
                
            hatu=self.u[i]/(1-self.beta1**self.t)
            hatv=self.v[i]/(1-self.beta2**self.t)
            self.param[i].cachedData=self.param[i].cachedData-self.lr*hatu/(hatv**0.5+self.ep)