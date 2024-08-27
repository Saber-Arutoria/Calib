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
            
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return opr.divide(opr.exp(x), opr.add_scalar(opr.exp(x), 1))
        
def sigmoid(x):
    return Sigmoid()(x)
    
class RNNCell(Module):
    def __init__(self,input_size,hidden_size,bias=True,dtype="float32"):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.W_ih=Parameter(init.xavier_uniform(1,1,shape=(self.input_size,self.hidden_size),dtype="float32"))
        self.W_hh=Parameter(init.xavier_uniform(1,1,shape=(self.hidden_size,self.hidden_size),dtype="float32"))
        self.ifbias=bias
        if bias==True:
            k=np.sqrt(1/self.hidden_size)
            self.b_ih=Parameter(init.unif(self.hidden_size,left=-k,right=k,dtype="float32"))
            self.b_hh=Parameter(init.unif(self.hidden_size,left=-k,right=k,dtype="float32"))
    
    def forward(self,x,h=None):
        xw=opr.matmul(x, self.W_ih)
        if h==None:
            if self.ifbias==True:
                r=opr.add(opr.add(xw,opr.broadcast_to(self.b_ih, (xw.shape))), opr.broadcast_to(self.b_hh, (xw.shape)))
            else:
                r=xw
                
        else:
            if self.ifbias==True:
                r=opr.add(opr.add(xw,opr.broadcast_to(self.b_ih, (xw.shape))), opr.add(opr.matmul(h,self.W_hh),opr.broadcast_to(self.b_hh, (xw.shape))))
                
            else:
                r=opr.add(xw,opr.matmul(h, self.W_hh))
        a=opr.tanh(r)
        return a
                
        
class RNN(Module):
    def __init__(self,input_size,hidden_size,num_layers=1,bias=True,dtype="float32"):
        super().__init__()
        self.ifbias=bias
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.rnncell=[]
        for i in range(0,self.num_layers):
            if i==0:
                self.rnncell.append(RNNCell(self.input_size, self.hidden_size,self.ifbias))
            else:
                self.rnncell.append(RNNCell(self.hidden_size, self.hidden_size,self.ifbias))
        
    def forward(self,X,h0=None):
        XL=opr.split(X, X.shape[0], 0)
        X1=[]
        for i in range(0,len(XL)):
            a=opr.tensor_tuple_getitem(XL, i)
           
            X1.append(a)
        XL=X1
        if h0==None:
            h0=init.zeros(*(self.num_layers,X.shape[1],self.hidden_size))
        hL=opr.split(h0, h0.shape[0], 0)
        HL=[]
        for i in range(0,len(hL)):
            a=opr.tensor_tuple_getitem(hL, i)
            
            HL.append(a)
        hL=HL
        y=[]
        for t in range(0,X.shape[0]):
            for l in range(0,self.num_layers):
                
                if l==0:
                   
                    hL[l]=self.rnncell[0].forward(XL[t],hL[l])
                    
                else:
                    hL[l]=self.rnncell[0].forward(hL[l-1],hL[l])
            y.append(hL[self.num_layers-1])
        
        feature=opr.cat(y, axes=0)
        
        hn=opr.stack(hL, axes=0)
        return feature,hn   
        
    
class LSTMCell(Module):
    def __init__(self,input_size,hidden_size,bias=True,dtype="float32"):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.W_ih=Parameter(init.xavier_uniform(1,1,shape=(self.input_size,4*self.hidden_size),dtype="float32"))
        self.W_hh=Parameter(init.xavier_uniform(1,1,shape=(self.hidden_size,4*self.hidden_size),dtype="float32"))
        self.ifbias=bias
        if bias==True:
            k=np.sqrt(1/self.hidden_size)
            self.b_ih=Parameter(init.unif(4*self.hidden_size,left=-k,right=k,dtype="float32"))
            self.b_hh=Parameter(init.unif(4*self.hidden_size,left=-k,right=k,dtype="float32"))
    
    def forward(self,x,h=None):
        if h==None:
            h0=init.zeros(*(x.shape[0],self.hidden_size))
            c0=init.zeros(*(x.shape[0],self.hidden_size))
            h=(h0,c0)
        h0=h[0]
        c0=h[1]
        xw=opr.matmul(x, self.W_ih)
        if self.ifbias==True:
            r=opr.add(opr.add(xw,opr.broadcast_to(self.b_ih, (xw.shape))), opr.add(opr.matmul(h0,self.W_hh),opr.broadcast_to(self.b_hh, (xw.shape))))
                
        else:
            r=opr.add(xw,opr.matmul(h0, self.W_hh))
        
        i,f,g,o=opr.split(r, 4, 1)
        i=sigmoid(i)
        f=sigmoid(f)
        g=opr.tanh(g)
        o=sigmoid(o)
        c1=opr.add(opr.multiply(f, c0),opr.multiply(i, g))
        h1=opr.multiply(o, opr.tanh(c1))
        return (h1,c1)
    
    
class LSTM(Module):
    def __init__(self,input_size,hidden_size,num_layers=1,bias=True,dtype="float32"):
        super().__init__()
        self.ifbias=bias
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstmcell=[]
        for i in range(0,self.num_layers):
            if i==0:
                self.lstmcell.append(LSTMCell(self.input_size, self.hidden_size,self.ifbias))
            else:
                self.lstmcell.append(LSTMCell(self.hidden_size, self.hidden_size,self.ifbias))
    def forward(self,X,h=None):
        XL=opr.split(X, X.shape[0], 0)
        X1=[]
        for i in range(0,len(XL)):
            a=opr.tensor_tuple_getitem(XL, i)
            a=opr.reshape(a,a.shape[1:])
            X1.append(a)
        XL=X1
        
        if h==None:
            h0=init.zeros(*(self.num_layers,X.shape[1],self.hidden_size))
            c0=init.zeros(*(self.num_layers,X.shape[1],self.hidden_size))
            h=(h0,c0)
        
        hL=h[0]
        hL=opr.split(hL,hL.shape[0] , 0)
        HL=[]
        for i in range(0,len(hL)):
            a=opr.tensor_tuple_getitem(hL, i)
            a=opr.reshape(a,a.shape[1:])
            HL.append(a)
        hL=HL
        cL=h[1]
        cL=opr.split(cL,cL.shape[0], 0)
        CL=[]
        for i in range(0,len(cL)):
            a=opr.tensor_tuple_getitem(cL, i)
            a=opr.reshape(a,a.shape[1:])
            CL.append(a)
        cL=CL
        y=[]
        for t in range(0,X.shape[0]):
            for l in range(0,self.num_layers):
                
                
                if l==0:
                    hL[l],cL[l]=self.lstmcell[0].forward(XL[t],(hL[l],cL[l]))
                else:
                    hL[l],cL[l]=self.lstmcell[l].forward(hL[l-1],(hL[l],cL[l]))
                               
            y.append(hL[self.num_layers-1])
        feature=opr.stack(y, axes=0)
        h0=opr.stack(hL, axes=0)
        c0=opr.stack(cL, axes=0)
        return (feature,(h0,c0))
        
        
class Embedding(Module):
    def __init__(self,num_embeddings,embedding_dim,dtype="float32"):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.W=Parameter(init.normal(*(self.num_embeddings,self.embedding_dim)))
        
    def forward(self,x):
        x1=init.one_hot(self.num_embeddings, x)
        return opr.matmul(x1, self.W)
        
        
class MultiHeadAttention(Module):
    def __init__(self,*,dropout=0.,causal=False,dtype="float32"):
        super().__init__()
        self.dtype=dtype
        self.causal=causal
        self.dropout=Dropout(dropout)
        
    def create_casual_mask(self,i,j):
        mask=-np.finfo(np.float32).max* np.triu(np.ones((1,1,i,j),dtype=np.float32),j-i+1)
        return mask
    
    def matmul(self,a,b_t):
        
        return opr.matmul(a, b_t)
    
    
    def softmax(self,z):
        max_val=Tensor(z.compute_value().max(axis=3),dtype=z.dtype,require_grad=False)
        max_val=opr.reshape(max_val, (*(z.shape[:-1]),1))
        max_val=opr.broadcast_to(max_val, z.shape)
        
        prob=opr.exp(opr.add(z, opr.negate(max_val)))
        denominator=opr.summation(prob,axes=(3,))
        
        denominator=opr.reshape(denominator, (*(z.shape[:-1]),1))
        denominator=opr.broadcast_to(denominator, z.shape)
        
        return opr.divide(prob, denominator)
    
    def forward(self,q,k,v):
        batch_size,num_head,queries_len,q_dim=q.shape
        _,_,key_value_len,k_dim=k.shape
        _,_,_,v_dim=v.shape
        
        assert q_dim==k_dim==v_dim
        
        result=None
        prob=None
        if self.causal==False:
            mask=Tensor(init.zeros(*(batch_size,num_head,queries_len,queries_len)))
        else:
            mask=Tensor(self.create_casual_mask(queries_len, queries_len))
            
        prob=self.dropout(self.softmax(opr.add(opr.divide_scalar(self.matmul(q, opr.transpose(k)), np.sqrt(v_dim)),mask)))
        result=self.matmul(prob, v)
        
        
        return result,prob
        
        
class AttentionLayer(Module):
    def __init__(self,q_features:int,num_head,dim_head,*,k_features:int=None,v_features:int=None,out_features:int=None,dropout=0.,causal=True,dtype="float32"):
        super().__init__()
        
        self.dtype=dtype
        if k_features is None:
            k_features=q_features
        
        if v_features is None:
            v_features=q_features
        
        if out_features is None:
            out_features=q_features
        
        self.q_features=q_features
        self.k_features=k_features
        self.v_features=v_features
        self.out_features=out_features
        self.num_head=num_head
        self.dim_head=dim_head
        
        self.prenorm_q=LayerNorm1d(q_features,dtype=self.dtype)
        self.prenorm_k=LayerNorm1d(k_features,dtype=self.dtype)
        self.prenorm_v=LayerNorm1d(v_features,dtype=self.dtype)
        
        inner_dim=num_head*dim_head
        
        
        self.q_projection=Linear(q_features,inner_dim,bias=False,dtype=dtype)
        self.k_projection=Linear(k_features,inner_dim,bias=False,dtype=dtype)
        self.v_projection=Linear(v_features,inner_dim,bias=False,dtype=dtype)
        
        self.attn=MultiHeadAttention(dropout=dropout,causal=causal,dtype=dtype)
        
        self.out_proj=Linear(inner_dim, out_features)
        
    def forward(self,q,k=None,v=None):
        if k is None:
            k=q
        if v is None:
            v=q
        batch_size,queries_len,q_dim=q.shape
        _,key_value_len,k_dim=k.shape
        _,_,v_dim=v.shape
        
        
        
        Q=self.q_projection(self.prenorm_q(q))
        K=self.k_projection(self.prenorm_k(k))
        V=self.v_projection(self.prenorm_v(v))
        
        Q=opr.reshape(Q, (Q.shape[0],Q.shape[1],self.num_head,self.dim_head))
        V=opr.reshape(V, (V.shape[0],V.shape[1],self.num_head,self.dim_head))
        K=opr.reshape(K, (K.shape[0],K.shape[1],self.num_head,self.dim_head))
        
        Q=opr.transpose(Q,(1,2))
        K=opr.transpose(K,(1,2))
        V=opr.transpose(V,(1,2))
        
        X,porb=self.attn(Q,K,V)
        X=opr.transpose(X,(1,2))
        X=opr.reshape(X, (X.shape[0],X.shape[1],self.num_head*self.dim_head))
        
        result=self.out_proj(X)
        
        
        return result
        
            
        
class TransformerLayer(Module):
    def __init__(self,q_features:int,num_head,dim_head,hidden_szie:int,*,dropout=0.,dtype="folat32"):
        super().__init__()
        self.dtype=dtype
        self.q_features=q_features
        self.hidden_size=hidden_szie
        
        self.atten_encoder=AttentionLayer(q_features, num_head, dim_head,dropout=dropout,causal=False,dtype=dtype)
        self.atten_decoder=AttentionLayer(q_features, num_head, dim_head,dropout=dropout,causal=True,dtype=dtype)
        
        self.L1=Linear(self.q_features,self.hidden_size)
        self.L2=Linear(self.hidden_size,self.q_features)      
        self.norm=LayerNorm1d(self.q_features,dtype=dtype)
        
        self.L3=Linear(self.q_features,self.hidden_size)
        self.L4=Linear(self.hidden_size,self.q_features)    
        self.dropout=dropout
    def forward(self,x):
        "Self-Attention and Feed Forward"
        
        x1=Residual(Sequent(self.atten_encoder,Dropout(self.dropout)))(x)
        x2=Residual(Sequent(self.norm,self.L1,ReLU(),Dropout(self.dropout),self.L2,Dropout(self.dropout)))(x1)

        kv=self.norm(x2)
        
        q1=self.norm(opr.add(x,Dropout(self.dropout)(self.atten_decoder(x))))
        
        y2=self.norm(opr.add(q1,Dropout(self.dropout)(self.atten_encoder(q1,kv,kv))))
        y3=Residual(Sequent(self.L3,ReLU(),Dropout(self.dropout),self.L4,Dropout(self.dropout)))(y2)
        y3=self.norm(y3)
       
        return y3
    
    
class Transformer(Module):
    def __init__(self,embedding_size,hidden_size,*,num_head,dim_head,dropout=0.,dtype="float32",sequence_len):
        super().__init__()
        self.emb=Embedding(sequence_len, embedding_size)
        self.trans=TransformerLayer(embedding_size,num_head,dim_head,hidden_size,dropout=dropout,dtype=dtype)
        self.proj=Linear(embedding_size, sequence_len)
        
        
    def softmax(self,z):
        max_val=Tensor(z.compute_value().max(axis=2),dtype=z.dtype,require_grad=False)
        max_val=opr.reshape(max_val, (*(z.shape[:-1]),1))
        max_val=opr.broadcast_to(max_val, z.shape)
        
        prob=opr.exp(opr.add(z, opr.negate(max_val)))
        denominator=opr.summation(prob,axes=(2,))
        
        denominator=opr.reshape(denominator, (*(z.shape[:-1]),1))
        denominator=opr.broadcast_to(denominator, z.shape)
        
        return opr.divide(prob, denominator)
    
        
    def forward(self,x):
        x=self.emb(x)
        x=self.trans(x)
        x=self.proj(x)
        
        x=self.softmax(x)
        
        return x
        
