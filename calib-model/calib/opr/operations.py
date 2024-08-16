# -*- coding: utf-8 -*-

from calib.CGF import Tensor,Tensor_Operation
from numbers import Number
import numpy as np

from typing import Tuple,List,Optional,Callable,Any
from calib import init

class Elewise_add(Tensor_Operation):
    def forward(self,a,b):
        return a+b
    
    def gradient(self, out_grad, node):
        return out_grad,out_grad
    
def add(a,b):
    return Elewise_add()(a,b)

class Add_Sca(Tensor_Operation):
    def __init__(self,c):
        self.c=c
    
    def forward(self, a):
        return a+self.c
    
    def gradient(self, out_grad, node):
        return out_grad
        
def add_scalar(a,c):
    return Add_Sca(c)(a)


class Elewise_mul(Tensor_Operation):
    def forward(self, a,b):
        return a*b
    
    def gradient(self, out_grad, node):
        c,d=node.input_val
        return multiply(out_grad,d), multiply(out_grad,c)
    
def multiply(a,b):
    return Elewise_mul()(a,b)


class mul_Sca(Tensor_Operation):
    def __init__(self,c):
        self.c=c
    
    def forward(self, a):
        return a*self.c
    
    def gradient(self, out_grad, node):
        return mul_scalar(out_grad,self.c)
    
def mul_scalar(a,c):
    return mul_Sca(c)(a)

class power_Sca(Tensor_Operation):
    def __init__(self,c):
        self.c=c
    
    def forward(self, a):
        return a**self.c
    
    def gradient(self, out_grad, node):
        r=node.input_val[0]
        
        return mul_scalar(multiply(out_grad,power_scalar(r,self.c-1)),self.c)
    
def power_scalar(a,c):
    
   
    return power_Sca(c)(a)
    
class Elewise_pow(Tensor_Operation):
    def forward(self, a,b):
        return a**b
    
    def gradient(self, out_grad, node):
        
        a,b=node.input_val[0],node.input_val[1]
        return multiply(out_grad,multiply(b,(power(a,add_scalar(b,-1))))),multiply(out_grad,multiply(power(a,b),log(a)))
    
def power(a,b):
    return Elewise_pow()(a,b)
    
    
class Elewise_divide(Tensor_Operation):
    def forward(self, a,b):
        return a/b
    
    def gradient(self, out_grad, node):
        a,b=node.input_val[0],node.input_val[1]
        return divide(out_grad,b),negate(multiply(out_grad,divide(a,power_scalar(b,2))))
    
def divide(a,b):
    return Elewise_divide()(a,b)

class Divide_sca(Tensor_Operation):
    def __init__(self,c):
        self.c=c
    
    def forward(self, a):
        return a/self.c
    
    def gradient(self, out_grad, node):
        return divide_scalar(out_grad, self.c)
    
def divide_scalar(a,c):
    return Divide_sca(c)(a)

class T(Tensor_Operation):
    def __init__(self,axes:Optional[tuple]=None):
        self.axes=axes
    
    def forward(self, a):
        if self.axes==None:
            return np.moveaxis(a,len(a.shape)-2,len(a.shape)-1)
        else:
            return np.moveaxis(a,self.axes[0],self.axes[1])
    
    def gradient(self, out_grad, node):
        return T(self.axes)(out_grad)
    
def transpose(a,axes=None):
    return T(axes)(a)
    
    
class Reshape(Tensor_Operation):
    def __init__(self,shape):
        self.shape=shape
        self.shape1=None
    def forward(self, a):
        self.shape1=a.shape
        
        return np.reshape(a, self.shape)
    def gradient(self, out_grad, node):
        return Reshape(self.shape1)(out_grad)
    
def reshape(a,shape):
    return Reshape(shape)(a)

class Broadcastto(Tensor_Operation):
    def __init__(self,shape):
        self.shape=shape
        self.shape1=None
    def forward(self, a):
        self.shape1=a.shape
        
        return np.broadcast_to(a, self.shape)
    def gradient(self, out_grad, node):
        s=node.input_val[0].shape
        s1=out_grad.shape
        
        s=list(s)
        t=[]
        for i in range(0,len(s1)):
          t.append(1)  
        for j in range(len(s)-1,-1,-1):
            
            t[j+len(t)-len(s)]=s[j]
            
        
        axis=[]
        for i in range(0,len(s1)):
            if s1[i] !=t[i] :
                axis.append(i)
        
        axis=tuple(axis)
        
        e=Summation(axis)(out_grad)
        
        q=Reshape(self.shape1)(e)
        
        return q
    
def broadcast_to(a, shape):
    return Broadcastto(shape)(a)


class Summation(Tensor_Operation):
    def __init__(self,axes:Optional[tuple]=None):
        self.axes=axes
        self.shape=None
    def forward(self, a):
        
        self.shape=a.shape
        return np.sum(a,self.axes)
    
    def gradient(self, out_grad, node):
        tup=list(node.input_val[0].shape)
        
        if self.axes!=None:
            
            for item in list(self.axes):
                tup[item]=1
            tup=tuple(tup)
            
            r=reshape(out_grad,tup)
            e=broadcast_to(r, node.input_val[0].shape)
        else:
            
            c=np.ones(node.input_val[0].shape)
            c=mul_scalar(out_grad, c)
            e=c
           
        return e
  
def summation(a,axes=None):
    return Summation(axes)(a)

class Matmul(Tensor_Operation):
    def forward(self, a,b):
        return np.matmul(a,b)
    
    def gradient(self, out_grad, node):
        a=node.input_val[0]
        b=node.input_val[1]
        
        grad_a=matmul(out_grad,T()(b))
        
        grad_b=matmul(T()(a),out_grad)
        tpa=[i for i in range(0,len(out_grad.shape)-len(a.shape))]
        tpb=[i for i in range(0,len(out_grad.shape)-len(b.shape))]
        tpa=tuple(tpa)
        tpb=tuple(tpb)
       
        if len(tpa)>0:
            grad_a=Summation(tpa)(grad_a)
        if len(tpb)>0:
            grad_b=Summation(tpb)(grad_b)
        
        
        return grad_a,grad_b
    
def matmul(a,b):
    return Matmul()(a,b)

class oppo(Tensor_Operation):
    def forward(self, a):
        return -a
    
    def gradient(self, out_grad, node):
        return mul_scalar(out_grad,-1)
    
def negate(a):
    return oppo()(a)

class LOG(Tensor_Operation):
    def forward(self, a):
        
        return np.log(a)

    def gradient(self, out_grad, node):
        return divide(out_grad,node.input_val[0])

def log(a):
    return LOG()(a)

class EXP(Tensor_Operation):
    def forward(self, a):
        return np.exp(a)
    
    def gradient(self, out_grad, node):
        return multiply(exp(node.input_val[0]),out_grad)

def exp(a):
    return EXP()(a)

class ReLu(Tensor_Operation):
    def forward(self, a):
        return (a+np.abs(a))/2
    
    def gradient(self, out_grad, node):
        r=node.to_numpy()
        c=np.sign(r)
        shape=c.shape
        c=np.reshape(c,(1,-1))
        for i in range(0,len(c[0])):
            if c[0][i]<0:
                c[0][i]=0
        c=np.reshape(c, shape)
        c=Tensor(c)
        c=multiply(out_grad,c)
        return c

def relu(a):
    return ReLu()(a)



class LogSumExp(Tensor_Operation):
    def __init__(self,axes:Optional[tuple]=None):
        self.axes=axes
        
    def forward(self, X):
        r=np.amax(X,self.axes,keepdims=True)
        r1=np.amax(X,self.axes)
        p=np.sum(np.exp(X-r),self.axes)
        return np.log(p)+r1
        
    def gradient(self, out_grad, node):
        Z=node.input_val[0].to_numpy()
    
        
        r1=np.amax(Z,self.axes,keepdims=True)
        r=np.exp(Z-r1)
        y=np.sum(r,axis=self.axes,keepdims=True)
        o=r/y
        
        o=Tensor(o)
        
        tup=list(node.input_val[0].shape)
        if self.axes!=None:
            
            for item in list(self.axes):
                tup[item]=1
            tup=tuple(tup)
            
            r=reshape(out_grad,tup)
            e=broadcast_to(r, node.input_val[0].shape)
            e=multiply(e, o)
            return e
        else:
            sum1=power_scalar(summation(exp(node.input_val[0])),-1)
            c=multiply(sum1,init.ones(*(node.input_val[0].shape)))
            c=multiply(exp(node.input_val[0]),c)
            c=multiply(out_grad,c)
            return c

def logsumexp(a,axes):
    return LogSumExp(axes)(a)



















