# -*- coding: utf-8 -*-

import numpy as np
import init
from typing import Tuple,List,Optional
import operations as opr
from operator import add
from functools import reduce    


num_tensor=0

class Operation:
    def __call__(self,*args):
        raise NotImplementedError()
    
    def forward(self,*args:Tuple[np.array]):
        raise NotImplementedError()
        
    def gradient(self,out_grad:"Element",node:"Element"):
        raise NotImplementedError()
        
    def convert_grad_to_tuple(self,out_grad:"Element",node:"Element"):
        out=self.gradient(out_grad, node)
        if isinstance(out, tuple):
            return out
        elif isinstance(out, list):
            return tuple(out)
        else:
            return (out,)

class Tensor_Operation(Operation):
    def __call__(self,*args):
        
        return Tensor.copy_with_operation(self,args)

class TensorTuple_Operation(Operation):
    def __call__(self, *args):
        return TensorTuple.copy_with_operation(self,args)


class Element:
    op:Optional[Operation]
    input_val=List["Element"]
    cachedData=np.array
    require_grad=bool
    
    
    def compute_value(self):
        if self.cachedData is not None:
            return self.cachedData
        else:
           
            self.cachedData=self.op.forward(*[ele.compute_value() for ele in self.input_val])
            return self.cachedData
    
    
    
    def is_input(self):
        return self.op ==None
    
    def __delete__(self):
        global num_tensor
        num_tensor=num_tensor-1
        
    def _init_ele(self,op:Optional[Operation()],input_val=List["Tensor"],*,num_out:int=1,cachedData:List[object]=None,require_grad:Optional[bool]=None):
        global num_tensor
        num_tensor=num_tensor+1
        if require_grad==None:
            require_grad=any(node.require_grad for node in input_val)
        self.op=op
        self.input_val=input_val
        self.num_out=num_out
        self.cachedData=cachedData
        self.require_grad=require_grad
        
    @classmethod
    def copy(cls,data,*,require_grad=False):
        item=cls.__new__(cls)
        item._init_ele(None,[],cachedData=data,require_grad=require_grad)
        return item
    @classmethod
    def copy_with_operation(cls,op:Operation,input:List["Element"]):
        item=cls.__new__(cls)
        item._init_ele(op,input)
        if item.require_grad==False:
            return item.offgraph()
        else:
            item.compute_value()
            return item
    
    
    
class TensorTuple(Element):
    def __len__(self):
       
        return len(self.compute_value())
    
    def __getitem__(self,index):
        return opr.tensor_tuple_getitem(self,index)
    
    def tuple(self):
        return tuple([ele for ele in self])
    
    def __repr__(self):
        return "CGF.TensorTuple"+str(self.tuple())
    
    def __str__(self):
        return self.__repr__()
   
    def __add__(self,b):
        assert isinstance(b, TensorTuple)
        assert len(self)==len(b)
        return opr.maketensortuple(*[opr.add(self[i], b[i]) for i in range(0,len(b))])
        
    
    def offgraph(self):
        return TensorTuple.copy(self.compute_value())
    
class Tensor(Element):
    grad="Tensor"
    
    def __init__(self,
                 val,
                 *,
                 dtype=None,require_grad=True,**kwargs):
        if isinstance(val, Tensor):
            if dtype==None:
                dtype=val.dtype
            if dtype==val.dtype:
                cachedData=val.compute_value()
               
            else:
                cachedData=np.array(val.to_numpy(),dtype=dtype)
        else:
            
            cachedData=np.array(val,dtype=dtype)
            
        self._init_ele(None,[],cachedData=cachedData,require_grad=require_grad)
    def to_numpy(self):
        result=self.compute_value()
        return result
        
    @staticmethod
    def copy_with_operation(op:Operation,inputs:List["Element"]):
        tensor=Tensor.__new__(Tensor)
        
        tensor._init_ele(op,inputs)
        if tensor.require_grad==False:
            return tensor.offgraph()
        else:
            tensor.compute_value()
            return tensor
    @staticmethod
    def copy(data,require_grad=False):
        tensor=Tensor.__new__(Tensor)

        tensor._init_ele(None,
                        [],
                        cachedData=data
                        if not isinstance(data, Tensor)
                        else data.compute_value(),
                        require_grad=require_grad )
        return tensor
    
    @property
    def data(self):
        return self.offgraph()
    
    @data.setter
    def data(self,val):
        assert isinstance(val, Tensor)
        assert val.dtype==self.dtype
        self.cachedData=val.compute_value()
    
    def offgraph(self):
        
        return Tensor.copy(self.compute_value())

        
    @property
    def shape(self):
        
        return self.compute_value().shape
    @property
    def dtype(self):
        return self.compute_value().dtype
    
    def backward(self,out_grad=None):
        if out_grad==None:
           
            out_grad=(init.ones(*self.shape,dtype=self.dtype))
        else:
            out_grad=(out_grad)
        G_backward(self,out_grad)
        
        
    def __add__(self,other):
        if isinstance(other, Tensor):
            return opr.Elewise_add()(self,other)
        else:
            return opr.Add_Sca(other)(self)
    
    def mul(self,other):
        if isinstance(other, Tensor):
            return opr.Elewise_mul()(self,other)
        else:
            return opr.mul_Sca(other)(self)
    
    def power(self,other):
        if isinstance(other, Tensor):
            return opr.Elewise_pow()(self,other)
        else:
            return opr.power_Sca(other)(self)
    
    def sub(self,other):
        if isinstance(other, Tensor):
            return opr.Elewise_add()(self,opr.oppo()(other))
        else:
            return opr.Add_Sca((-1)*other)(self)
    
    def div(self,other):
        if isinstance(other, Tensor):
            return opr.Elewise_divide()(self,other)
        else:
            return opr.Divide_sca(other)(self)
    
    def matmul(self,other):
        return opr.Matmul()(self,other)
    
    def summation(self,axes=None):
        return opr.Summation(axes)(self)
    
    def broadcast_to(self,shape):
        return opr.Broadcastto(shape)(self)
    
    def reshape(self,shape):
        return opr.Reshape(shape)(self)
    
    def opp(self):
        return opr.oppo()(self)
    
    def transpose(self,axes=None):
        return opr.T(axes)(self)
    
    
    
def G_backward(tail,out_grad):
   
    output_start_from={}
    output_start_from[tail]=[out_grad]
    
    reverse_list=list(reversed(topo_sort([tail])))
    
    for t in range(0,len(reverse_list)):
        node=reverse_list[t]
        if isinstance(node, TensorTuple):
            grad_node=output_start_from[node][0]
            for k in range(1,len(output_start_from[node])):
                grad_node=add(grad_node,output_start_from[node][k] )
        else:
            
            grad_node=output_start_from[node][0]
            for k in range(1,len(output_start_from[node])):
                grad_node=opr.add(grad_node,output_start_from[node][k])
        node.grad=grad_node
        
        if len(node.input_val)!=0:
            dp=node.op.gradient(grad_node,node)
            
            if dp!=None:
                if type(dp)==tuple:
                    dp=list(dp)
                else:
                    dp=[dp]
                    
                for k in range(0,len(node.input_val)):
                    pred=node.input_val[k]
                    if pred in output_start_from:
                        output_start_from[pred].append(dp[k])
                    else:
                        output_start_from[pred]=[dp[k]]
                   
    
    
    
def topo_sort(tail_list):
    topo=[]
    gray=set()
    balck=set()
    def G_visit(node):
        
        if node in balck:
            return
        if node in gray:
            raise "Diag Error"
        gray.add(node)
        for pred in node.input_val:
            G_visit(pred)
        balck.add(node)
        topo.append(node)
        
    G_visit(tail_list[0])
    return topo
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    