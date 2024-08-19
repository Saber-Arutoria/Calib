# -*- coding: utf-8 -*-

import CGF 
import numpy as np
import math

def const(*shape,c=1.0,dtype="float32",require_grad=False):
    arr=c*np.ones(shape,dtype=dtype)
    return CGF.Tensor(arr,dtype=dtype,require_grad=require_grad)

def ones(*shape,dtype="float32",require_grad=False):
    return const(*shape,c=1.0,dtype=dtype,require_grad=require_grad)

def zeros(*shape,dtype="float32",require_grad=False):
    return const(*shape,c=0.0,dtype=dtype,require_grad=require_grad)

def unif(*shape,left=0.0,right=1.0,dtype="float32",require_grad=False):
    arr=np.random.rand(*shape)*(right-left)+left
    return CGF.Tensor(arr,dtype=dtype,require_grad=require_grad)

def normal(*shape,mean=0.0,std=1.0,dtype="float32",require_grad=False):
    arr=np.random.randn(*shape)*std+mean
    return CGF.Tensor(arr,dtype=dtype,require_grad=require_grad)
    
def benu(*shape,p=0.5,dtype="float32",require_grad=False):
    arr=np.random.rand(*shape)<=p
    return CGF.Tensor(arr,dtype=dtype,require_grad=require_grad)

def one_hot(n,i,dtype="float32",require_grad=False):
    arr=np.eye(n,dtype=dtype)[i.to_numpy()]
    return CGF.Tensor(arr,dtype=dtype,require_grad=require_grad)

def xavier_uniform(input_size,output_size,shape=None,e=1.0,**kwargs):
    if shape==None:
        
        a=e*math.sqrt(6/(input_size+output_size))
        r=unif(input_size,output_size,left=-a,right=a,**kwargs)
    else:
        a=e*math.sqrt(6/(input_size+output_size))
        r=unif(*shape,left=-a,right=a,**kwargs)
    return r
    
def xavier_normal(input_size,output_size,e=1.0,**kwargs):
    std=e*math.sqrt(2/(input_size+output_size))
    r=normal(input_size,output_size,mean=0.0,std=std,**kwargs)
    return r

