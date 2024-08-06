import numpy as np
import math
import torch


class buffer():
    #ER
    def __init__(self,mem_size=1000):
        self.mem_size=mem_size
        self.mem=[]
        self.item_len=None
        #print(type(self.mem))
    def is_full(self):
        if len(self.mem)<=self.mem_size:
            return True
        else:
            return False
    def is_empty(self):
        if len(self.mem):
            return False
        return True
    def update(self,x):
        cur_size=len(self.mem)
        #print(cur_size)
        if cur_size<self.mem_size:
            self.mem.append(x)
        else:
            #print(cur_size)
            rep_id=np.random.randint(0,len(self.mem))
            self.mem[rep_id-1]=x
    def data(self):
        return self.mem
    def len(self):
        return len(self.mem)
    def sample(self,random=True,idx=0):
        if random:
            idx=np.random.randint(0,len(self.mem))
        if self.is_empty():
            raise ValueError("buffer is empty")
        return self.mem[idx]
    def shape(self):
        return (len(self.mem),len(self.mem[0]))
    def empty(self):
        return self.mem.clear()
    