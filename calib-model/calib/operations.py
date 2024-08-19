# -*- coding: utf-8 -*-

from numbers import Number
import numpy as np
from CGF import Operation, Element,Tensor,Tensor_Operation,TensorTuple,TensorTuple_Operation
from typing import Tuple,List,Optional
import init
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
            return np.swapaxes(a,len(a.shape)-2,len(a.shape)-1)
        else:
            return np.swapaxes(a,self.axes[0],self.axes[1])
    
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


class Tanh(Tensor_Operation):
    def forward(self, a):
        return np.tanh(a)
    
    def gradient(self, out_grad, node):
        r=power_scalar(tanh(node.input_val[0]), 2)
        return multiply(negate(add_scalar(r, -1)),out_grad)

def tanh(x):
    return Tanh()(x)

class MakeTensorTuple(TensorTuple_Operation):
    def forward(self, *args):
        
        return tuple(args)
    
    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        
        return tuple([out_grad[i] for i in range(len(out_grad))])
    
def maketensortuple(*args):
    return MakeTensorTuple()(*args)

class TensorTupleGetItem(Tensor_Operation):
    def __init__(self,index):
        self.index=index
        
    def __call__(self,a:TensorTuple,fold_const=True):
        assert isinstance(a, TensorTuple)
        if fold_const==True and isinstance(a.op, MakeTensorTuple):
            return a.input_val[self.index]
        return Tensor.copy_with_operation(self,[a])
            
    def forward(self, a):
        return a[self.index]
    
    def gradient(self, out_grad, node):
        L=[]
        for k,val in enumerate(node.input_val[0]):
            if k!=self.index:
                L.append(init.zeros(*val.shape))
            else:
                L.append(out_grad)
        return MakeTensorTuple()(*L)
    
def tensor_tuple_getitem(a,index):
    return TensorTupleGetItem(index)(a)


class Cat(Tensor_Operation):
    def __init__(self,axes):
        self.axes=axes
    
    def forward(self, a:TensorTuple):
        L=[]
        for i in range(0,len(a)):
            self.shape=a[i].shape
            L.append(a[i])
        L=tuple(L)
   
        return np.concatenate(L,axis=self.axes)
    
    def gradient(self, out_grad, node):
        l=len(node.input_val[0])
        L=[]
        a=split(out_grad, l,self.axes)
        for i in range(0,l):
            r=tensor_tuple_getitem(a, i)
            L.append(r)
        return maketensortuple(*L)

def cat(args,axes):
    return Cat(axes)(maketensortuple(*args))

class Split(TensorTuple_Operation):
    def __init__(self,ind,axes):
        self.axes=axes
        self.ind=ind
    def forward(self, a:Tensor):
        L=np.split(a,self.ind,axis=self.axes)
        return L
    
    def gradient(self, out_grad, node):
        
        return Cat(self.axes)(out_grad)
    


def split(a,ind,axes):
    return Split(ind,axes)(a)

class Stack(Tensor_Operation):
    def __init__(self,axes):
        self.axes=axes
        self.shape=None
    def forward(self, a:TensorTuple):
        L=[]
        for i in range(0,len(a)):
            self.shape=a[i].shape
            L.append(a[i])
       
        return np.stack(L,self.axes)
    
    def gradient(self, out_grad, node):
        
        l=len(node.input_val[0])
        L=[]
        a=split(out_grad, l,self.axes)
        for i in range(0,l):
            p=tensor_tuple_getitem(a, i)
            r=reshape(p,self.shape)
            L.append(r)
        return maketensortuple(*L)
        

def stack(args,axes):
    return Stack(axes)(maketensortuple(*args))


class Flip(Tensor_Operation):
    def __init__(self,axes:Optional[tuple]=None):
        self.axes=axes
        
    def forward(self, a):
        return np.flip(a,self.axes)
    
    def gradient(self, out_grad, node):
        return flip(out_grad,self.axes)

def flip(a,axes):
    return Flip(axes)(a)
    
class Dilate(Tensor_Operation):
    def __init__(self,axes:tuple,d:int):
        self.axes=axes
        self.d=d
        
    def forward(self, a):
        t=sorted(self.axes)
        c=a
        for de in t:
            d=np.moveaxis(c, de, 0)
            shape=list(d.shape)
            shape[0]=shape[0]*(self.d+1)
            shape=tuple(shape)
            r=np.zeros(shape)
           
            for i in range(0,d.shape[0]):
                r[i*(self.d+1)]=d[i]
            c=np.moveaxis(r, 0, de)
            
        r=c
        return r
    
    def gradient(self, out_grad, node):
        
        return undilate(out_grad, self.axes, self.d)
    
def dilate(a,axes,d):
    return Dilate(axes, d)(a)


class Undilate(Tensor_Operation):
    def __init__(self,axes:tuple,d:int):
        self.axes=axes
        self.d=d
        
    def forward(self, a):
        t=sorted(self.axes)
        c=a
        for de in t:
            d=np.moveaxis(c, de, 0)
            shape=list(d.shape)
            shape[0]=int(shape[0]/(self.d+1))
            shape=tuple(shape)
            r=np.zeros(shape)
            for i in range(0,r.shape[0]):
                r[i]=d[i*(self.d+1)]
            c=np.moveaxis(r, 0, de)
        
        r=c
        return r
            
      
    
    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.d)
    

def undilate(a,axes,d):
    return Undilate(axes, d)(a)
    
    
class Conv(Tensor_Operation):
    def __init__(self,stride:Optional[int]=1,padding:Optional[int]=0):
        self.stride=stride
        self.padding=padding
        
    def forward(self, X,W):
        Q=np.pad(X, ((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        shape=(X.shape[0],int((Q.shape[1]-W.shape[0])/self.stride)+1,int((Q.shape[2]-W.shape[1])/self.stride)+1,W.shape[3])
        r=np.zeros(shape)
                    
           
        for i in range(0,W.shape[0]):
            for j in range(0,W.shape[1]):
                r=r+np.matmul(Q[:,i:i+r.shape[1]*self.stride:self.stride,j:j+r.shape[2]*self.stride:self.stride,:],W[i,j])
        return r
    def gradient(self, out_grad, node):
        grad=out_grad.offgraph()
        p=node.input_val[1].shape[0]-1
        grad=dilate(grad, (1,2), self.stride-1)
        grad=grad.offgraph().to_numpy()
        grad=np.pad(grad,((0,0),(p,p),(p,p),(0,0)))
        w=node.input_val[1].offgraph().to_numpy()
        w1=np.flip(w,(0,1))
        w1=np.moveaxis(w1, 2, 3)
        grad=Tensor(grad,require_grad=False)
        w1=Tensor(w1,require_grad=False)
        grad_x=conv(grad,w1,stride=1,padding=0)
        grad_x=grad_x.to_numpy()
        grad_x=grad_x[:,self.padding:self.padding+node.input_val[0].shape[1],self.padding:self.padding+node.input_val[0].shape[2],:]
        grad_x=Tensor(grad_x,require_grad=False)
        
        grad=out_grad.offgraph()
        n=node.input_val[1].shape[0]
        
        x1=node.input_val[0].offgraph().to_numpy()
        x1=np.pad(x1,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        x1=Tensor(x1)
        x1=transpose(x1,(3,0))
        m=x1.shape[1]
        grad1=transpose(grad,(0,1))
        grad1=transpose(grad1,(1,2))
        
        grad1=dilate(grad1,(0,1), self.stride-1)
        grad1=grad1.offgraph().to_numpy()
        grad1=grad1[:m-n+1,:m-n+1,:,:]
        grad1=Tensor(grad1)
        w=conv(x1, grad1)
        w=transpose(w,(0,1))
        w=transpose(w,(1,2))
        
        
        
        
        
        return grad_x,w 
        
    


def conv(x,w,stride=1,padding=0):
    return Conv(stride,padding)(x,w)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


