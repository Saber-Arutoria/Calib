# -*- coding: utf-8 -*-

import sys
import CGF 
import numpy as np
from numbers import Number
from typing import List,Callable,Any
import init
from operator import add
from functools import reduce  
import operations as opr 
import math
from CGF import Tensor

class Parameter(Tensor):
    "data tat can be learned"
    
def get_param(val:object):
    if isinstance(val, Parameter):
        return [val]
    elif isinstance(val, Module):
        return val.parameters()
    elif isinstance(val, dict):
        L=[]
        for k,l in val.items():
            L+=get_param(l)
        return L
    elif isinstance(val, (list,tuple)):
        L=[]
        for j in val:
            L+=get_param(j)
        return L
    
    else:
        return[]
    
def get_child_model(val:object):
    if isinstance(val, Module):
        L=[val]
        L.extend(get_child_model(val.__dict__))
        return L
    elif isinstance(val, dict):
        L=[]
        for k,l in val.items():
            L+=get_child_model(l)
        return L
    elif isinstance(val, (list,tuple)):
        L=[]
        for j in val:
            L+=get_child_model(j)
        return L
    else:
        return []
    
    
class Module:
    def __init__(self):
        self.training=True
        
    def parameters(self):
        return get_param(self.__dict__)
    
    def children_model(self):
        return get_child_model(self.__dict__)
    
    
    def run(self):
        self.training=False
        for c in self.children_model():
            c.training=False
            
    def train(self):
        self.training=True
        for c in self.children_model():
            c.training=True
        
    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)
        
        

class Linear(Module):
    def __init__(self,input_size,output_size,bias=True,dtype="float32"):
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.ifbias=bias
        self.weight=Parameter(init.xavier_uniform(self.input_size,self.output_size,dtype="float32"))
        if self.ifbias==True:
            e=math.sqrt(2)
            b=e*math.sqrt(3/self.output_size)
            self.bias=Parameter(init.unif(1,self.output_size,left=-b,right=b,dtype="float32"))

    def forward(self,X):
        row=X.shape[0]
        
        if self.ifbias==True:
            
            return opr.add(opr.matmul(X,self.weight),opr.broadcast_to(self.bias,(row,self.output_size)))
        else:
            
            return opr.matmul(X,self.weight)
        
class Flatten(Module):
    
    def forward(self,X):
        
        return opr.reshape(X, (X.shape[0],-1))
    
class ReLU(Module):
    def forward(self,X):
        return opr.relu(X)
    
class Sequent(Module):
    def __init__(self,*modules):
        super().__init__()
        
        self.modules=modules
        
    def forward(self,x):
        out=x
        for i in range(0,len(self.modules)):
            out=self.modules[i](out)
        return out
    


class MseLoss(Module):
    def forward(self,pred, y):
        return opr.divide_scalar(opr.summation(opr.power_scalar(opr.add(pred,opr.negate(y)),2)),pred.shape[0])
    
class SoftmaxLoss(Module):
    def forward(self,pred,y):
        y_one_hot=init.one_hot(pred.shape[1],y)
        L=opr.divide_scalar(opr.summation(opr.add(opr.logsumexp(pred,(1,)),opr.mul_scalar(opr.summation(opr.multiply(pred,y_one_hot),(1,)),-1))),pred.shape[0])
        return L
    

class BatchNorm1d(Module):
    def __init__(self,features,ep=1e-5,m=0.1,dtype="float32"):
        super().__init__()
        self.features=features
        self.ep=ep
        self.m=m
        self.weight=Parameter(init.ones(self.features))
        self.bias=Parameter(init.zeros(self.features))
        self.running_mean=init.zeros(self.features)
        self.running_var=init.ones(self.features)
        
    def forward(self,x):
        
        if self.training:
            
            Ex=opr.divide_scalar(opr.summation(x,(0,)),x.shape[0])
            self.running_mean=opr.add(opr.mul_scalar(self.running_mean,1-self.m),opr.mul_scalar(Ex,self.m))
           
            Ex=opr.reshape(Ex,(1,-1))
            
            Ex=opr.broadcast_to(Ex,x.shape)
          
            VarX=opr.divide_scalar(opr.summation(opr.power_scalar(opr.add(x,opr.negate(Ex)),2),(0,)),x.shape[0])
            self.running_var=opr.add(opr.mul_scalar(self.running_var,1-self.m),opr.mul_scalar(VarX,self.m))
            VarX=opr.reshape(VarX,(1,-1))
            
            VarX=opr.broadcast_to(VarX,x.shape)
            k=init.ones(*VarX.shape)
            
            
            x1=opr.divide(opr.add(x,opr.negate(Ex)),opr.power_scalar(opr.add(VarX,opr.mul_scalar(k,self.ep)),0.5))
            R=opr.reshape(self.weight,(1,-1))
            R=opr.broadcast_to(R,x1.shape)
            B=opr.reshape(self.bias,(1,-1))
            B=opr.broadcast_to(B,x1.shape)
            
            return opr.add(opr.multiply(R,x1),B)
        else:
           
          
           
            Ex=opr.reshape(self.running_mean,(1,-1))
            
            Ex=opr.broadcast_to(Ex,x.shape)
            
            
            VarX=opr.reshape(self.running_var,(1,-1))
            
            VarX=opr.broadcast_to(VarX,x.shape)
            k=init.ones(*VarX.shape)
            
            
            x1=opr.divide(opr.add(x,opr.negate(Ex)),opr.power_scalar(opr.add(VarX,opr.mul_scalar(k,self.ep)),0.5))
            R=opr.reshape(self.weight,(1,-1))
            R=opr.broadcast_to(R,x1.shape)
            B=opr.reshape(self.bias,(1,-1))
            B=opr.broadcast_to(B,x1.shape)
            return opr.add(opr.multiply(R,x1),B)

class BatchNorm2d(BatchNorm1d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    def forward(self, x):
        shape=x.shape
        x1=opr.transpose(x,(1,2))
        x1=opr.transpose(x1,(2,3))
        x1=opr.reshape(x1,(shape[0]*shape[2]*shape[3],shape[1]))
        y=opr.reshape(super().forward(x1),(shape[0],shape[2],shape[3],shape[1]))
        
        y=opr.transpose(y,(2,3))
        y=opr.transpose(y,(1,2))
        return y
        



class LayerNorm1d(Module):
    def __init__(self,features,ep=1e-5,dtype="float32"):
        super().__init__()
        self.features=features
        self.ep=ep
        self.weight=Parameter(init.ones(self.features))
        self.bias=Parameter(init.zeros(self.features))
       
    def forward(self,x):
        Ex=opr.divide_scalar(opr.summation(x,(1,)),x.shape[1])
        Ex=opr.reshape(Ex,(-1,1))
        Ex=opr.broadcast_to(Ex,x.shape)
        
        VarX=opr.divide_scalar(opr.summation(opr.power_scalar(opr.add(x,opr.negate(Ex)),2),(1,)),x.shape[1])
        VarX=opr.reshape(VarX,(-1,1))
        VarX=opr.broadcast_to(VarX,x.shape)
        k=init.ones(*VarX.shape)
        
        x1=opr.divide(opr.add(x,opr.negate(Ex)),opr.power_scalar(opr.add(VarX,opr.mul_scalar(k,self.ep)),0.5))
        R=opr.reshape(self.weight,(1,-1))
        R=opr.broadcast_to(R,x1.shape)
        B=opr.reshape(self.bias,(1,-1))
        B=opr.broadcast_to(B,x1.shape)
        return opr.add(opr.multiply(R,x1),B)

class Dropout(Module):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p
        
    def forward(self,x):
        if self.training and self.p!=0.0:   
            shape=x.shape
            r=init.benu(*shape)
            r=opr.mul_scalar(r,1/(1-self.p))
            y=opr.multiply(x,r)
            
            
            
            return y
        else:
            return x


class Residual(Module):
    def __init__(self,F):
        super().__init__()
        self.F=F
    
    def forward(self,x):
        return opr.add(self.F(x),x)
    
    
    
    
class Conv(Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,bias=True,dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size=kernel_size[0]
        if isinstance(stride, tuple):
            stride=stride[0]
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.ifboas=bias
        self.weight=Parameter(init.xavier_uniform(1, 1,shape=(self.kernel_size,self.kernel_size,self.in_channel,self.out_channel),nonlinearity="relu",dtype="float32"))
        if bias==True:
            b=1/(self.in_channel*self.kernel_size**2)**0.5
            self.bias=Parameter(init.unif(self.out_channel,left=-b,right=b,dtype="float32"))

    def forward(self,x):
        x=opr.transpose(x,(1,2))
        x=opr.transpose(x,(2,3))
        xw=opr.conv(x, self.weight,stride=self.stride,padding=self.kernel_size//2)
        
        
        if self.ifboas==True:
            a=opr.add(xw,opr.broadcast_to(self.bias,xw.shape ))
            a=opr.transpose(a,(1,3))
            a=opr.transpose(a,(2,3))
            return a
        else:
            return xw
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    