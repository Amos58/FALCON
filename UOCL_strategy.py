from deepod.metrics import ts_metrics,point_adjustment
import random
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans,KMeans
import torch.nn.functional as F
#from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario
from torch.utils.data import Dataset
from mem import buffer
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from typing import Any, Optional, Sequence, Union,Dict
#from avalanche.benchmarks.utils.classification_dataset import (SupportedDataset,make_classification_dataset,)
#from avalanche.benchmarks.utils.utils import concat_datasets_sequentially
import tent

from train_strategy import adjust_learning_rate


def test_with_navie_tta(test_loader,model,criterion,opt,threshold,ema,config,buffer_size=10000,batch_size=32):
    ema.apply_shadow()
    #results=[[] for i in range(194374)]
    attens_energy=[]
    test_labels=[]
    cnt=0
    lr=config.lr
    wd=config.wd
    if opt=="SGD":
        optim=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="AdamW":
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    for (x,y) in tqdm(test_loader):
        columns_num=x.shape[-1]
        padding_num=0 if columns_num%2==0 else 1
        x=F.pad(x,(0,padding_num),'constant',0)
        #cnt+=1
        x_rec=model(x.permute(0,2,1).cuda())
        #sign=True
        for u in range(x_rec.shape[0]):
            rec_loss=criterion(x_rec[u],x[u].float().cuda())
            cri=rec_loss.unsqueeze(0)
            cri=cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(y[u].unsqueeze(0))
            if cri<threshold:
                optim.zero_grad()
                loss=criterion(x.cuda(),x_rec.cuda())
                loss.backward()
                optim.step()
    #ts_metrics(labels,point_adjustment(labels,ans_1))
    attens_energy=np.concatenate(attens_energy,axis=0).reshape(-1)
    test_labels=np.concatenate(test_labels,axis=0).reshape(-1)
    attens_energy=np.array(attens_energy)
    test_labels=np.array(test_labels)
    return ts_metrics(test_labels,point_adjustment(test_labels,attens_energy)),attens_energy,test_labels


def train_ER_(model,his,x,y,criterion,opt):
    his_y=model(his.cuda())
    l1=criterion(his_y,his.cuda())
    l2=criterion(y.cuda(),x.cuda())
    l=0.3*l1+0.7*l2
    opt.zero_grad()
    l.backward()
    opt.step()
    return model



def test_with_er_tta(test_loader,model,criterion,opt,threshold,ema,config,buffer_size=10000,batch_size=32):
    ema.apply_shadow()
    mem=buffer(buffer_size)
    attens_energy=[]
    test_labels=[]
    lr=config.lr
    wd=config.wd
    if opt=="SGD":
        optim=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="AdamW":
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    for x,y in tqdm(test_loader):
        columns_num=x.shape[-1]
        padding_num=0 if columns_num%2==0 else 1
        x=F.pad(x,(0,padding_num),'constant',0)
        x=x.float().cuda()
        x_rec=model(x.permute(0,2,1))#.float().cuda())
        #l=criterion(y,x)
        for u in range(x_rec.shape[0]):
            rec_loss=criterion(x,x_rec)
            cri=rec_loss.unsqueeze(0)
            cri=cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(y[u].unsqueeze(0))
            if cri<threshold:
                mem.update(x.detach().cpu())
                his=mem.sample()
                model=train_ER(model,his,x,criterion,opt)
    #ts_metrics(labels,point_adjustment(labels,ans_1))
    attens_energy=np.concatenate(attens_energy,axis=0).reshape(-1)
    test_labels=np.concatenate(test_labels,axis=0).reshape(-1)
    attens_energy=np.array(attens_energy)
    test_labels=np.array(test_labels)
    return ts_metrics(test_labels,point_adjustment(test_labels,attens_energy)),attens_energy,test_labels

def threshold_update(threshold,pred,y,interval=1e-4):
    if pred==0 and y==1:
        threshold=threshold-interval*3
    elif pred==1 and y==0:
        threshold=threshold+interval*3
    else:
        threshold=threshold+interval
    return threshold


def test_with_dynamic_threshold(test_loader,model,label,criterion,opt,threshold,buffer_size=10000):
    mem=buffer(buffer_size)
    results=[[] for i in range(len(test_loader))]
    for x,y in tqdm(test_loader):
        x=x.cuda()
        pred=model(x)
        l=criterion(pred,x)
        L=1
        if l.detach().cpu().item() <threshold:
            mem.update(x.detach().cpu())
            his=mem.sample()
            model=train_ER(model,his,x,criterion,opt)
            L=0
        threshold=threshold_update(threshold,L,y)
        tmp_x=x.detach().cpu().view(-1,38)
        tmp_y=y.detach().cpu().view(-1,38)
        for i in range(0,tmp_y.shape[0]):
            results[cnt+i].append(criterion(tmp_y[i],tmp_x[i]).item())
        cnt+=1
    ans=[]
    for i in results:
        ans.append(np.mean(i))
    ans=np.array(ans)
    np.nan_to_num(ans,0)
    ans_1=ans[:len(label)]
    #ts_metrics(labels,point_adjustment(labels,ans_1))
    return ts_metrics(label,point_adjustment(label,ans_1))


def train_Kmeans_(model,mem,reservoir,x,y,criterion,opt,config,batch_size=32):
    reservoir.update(x)
    lr=config.lr
    wd=config.wd
    if opt=="SGD":
        optim=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="AdamW":
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    if reservoir.is_full():
        best_scores=-1
        best_centroids=np.array([])
        for i in range(2,reservoir.len()//10):
            kmeans=KMeans(n_clusters=i,random_state=42)
            results=kmeans.fit(reservoir.data)
            scores=silhouette_score(reservoir.data,results.labels_)
            if scores>best_scores:
                best_centroids=results.cluster_centers_
        for i in range(len(best_centroids)):
            mem.update(best_centroids[i])
    his_r,his_k=[],[]#(batch,features)
    sign=False
    for i in range(max(batch_size,max(mem.len(),reservoir.len()))):
        his_r.append(reservoir.sample())
        if mem.len()==0:
            continue
        else:
            sign=True
            his_k.append(mem.sample())
    opt.zero_grad()
    his_r=torch.stack(his_r).view(batch_size,x.shape[1],x.shape[2])
    #print(type(his_r))
    #print(his_r.shape)
    #print(type(his_r[0]))
    #print(his_r[0])
    pred_r=model(his_r.cuda())
    l1=criterion(x.cuda(),y.cuda())
    l3=criterion(his_r.cuda(),pred_r.cuda())
    if sign:
        pred_k=model(his_k)
        l2=criterion(his_k.cuda(),pred_k.cuda())
        l=0.7*l1+0.2*l3+0.1*l2
        l.backward()    
    else:
        l=0.8*l1+0.2*l3
        l.backward()
    opt.step()
    if reservoir.is_full():
            reservoir.empty()
    return model,mem,reservoir

def test_with_kmeans_tta(test_loader,model,criterion,opt,threshold,ema,config,buffer_size=10000,batch_size=32):
    mem=buffer(buffer_size)
    reservoir=buffer(buffer_size) 
    lr=config.lr
    wd=config.wd
    if opt=="SGD":
        optim=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="AdamW":
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    cnt=0
    ema.apply_shadow()
    attens_energy=[]
    test_labels=[]
    #kmeans=MiniBatchKMeans(n_clusters=)
    model=model.cuda()
    for x,y in tqdm(test_loader):
        x=x.float().cuda()
        columns_num=x.shape[-1]
        padding_num=0 if columns_num%2==0 else 1
        x=F.pad(x,(0,padding_num),'constant',0)
        rec_x=model(x.permute(0,2,1))
        for u in range(rec_x.shape[0]):
            rec_loss=criterion(rec_x[u],x[u])
            cri=rec_loss.unsqueeze(0)
            cri=cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(y[u].unsqueeze(0))
            if cri<threshold:
                model,mem,reservoir=train_Kmeans_(model,mem,reservoir,x,y,criterion,optim,config,batch_size)#(model,mem,reservoir,x,y,criterion,opt,config,batch_size=32)
    attens_energy=np.concatenate(attens_energy,axis=0).reshape(-1)
    test_labels=np.concatenate(test_labels,axis=0).reshape(-1)
    attens_energy=np.array(attens_energy)
    test_labels=np.array(test_labels)
    return ts_metrics(test_labels,point_adjustment(test_labels,attens_energy)),attens_energy,test_labels
def criterion(y_pred,y_true,log_vars):
    loss=0
    for i in range(len(y_pred)):
        precision=torch.exp(-log_vars[i])
        diff=(y_pred[i]-y_true[i])**2
        loss+=torch.sum(precision*diff+log_vars[i],-1)
    return torch.mean(loss)

def train_Kmeans(model,mem,reservoir,x,y,criterion,opt,ema,config,log_var_a,log_var_b,log_var_c,batch_size=32):
    reservoir.update(x)
    lr=config.lr

    wd=0#config.wd
    if reservoir.is_full():
        best_scores=-1
        best_centroids=np.array([])
        for i in range(2,reservoir.len()//10):
            kmeans=KMeans(n_clusters=i,random_state=42)
            results=kmeans.fit(reservoir.data)
            scores=silhouette_score(reservoir.data,results.labels_)
            if scores>best_scores:
                best_centroids=results.cluster_centers_
        for i in range(len(best_centroids)):
            mem.update(best_centroids[i])
   # his_r,his_k#(batch,features)
    sign=False
    for i in range(max(batch_size,max(mem.len(),reservoir.len()))):
        his_r=reservoir.sample()
        if mem.len()==0:
            continue
        else:
            sign=True
            his_k=mem.sample()
    opt.zero_grad()

    pred_r=model(his_r.cuda().permute(0,2,1))
    #print("x,y",x,y)
    #l1=criterion(x.float(),y.float())

    if sign==False:
        his_k=torch.zeros_like(his_r).cuda()
        pred_k=model(his_k.permute(0,2,1))
        #l2=criterion(his_k.cuda(),pred_k.cuda())
        #l=config.alpha*l1+0.2*l3+0.1*l2
        #l.backward(retain_graph=True)    
    #print(pred_k.device,pred_r.device,y.device,his_k.device,his_r.device,x.device)
    
    loss=criterion([pred_k,pred_r,y],[his_k,his_r,x],[log_var_a,log_var_b,log_var_c])
    
    loss.backward()
    opt.step()
    ema.update()
    if reservoir.is_full():
            reservoir.empty()
    return model,mem,reservoir



def train_ER_(model,reservoir,x,y,criterion,opt,ema,config,log_var_a,log_var_b,batch_size=32):
    reservoir.update(x)
    lr=config.lr

    wd=0#config.wd
    """
    if reservoir.is_full():
        best_scores=-1
        #best_centroids=np.array([])
        for i in range(2,reservoir.len()//10):
            kmeans=KMeans(n_clusters=i,random_state=42)
            results=kmeans.fit(reservoir.data)
            scores=silhouette_score(reservoir.data,results.labels_)
            if scores>best_scores:
                best_centroids=results.cluster_centers_
        for i in range(len(best_centroids)):
            mem.update(best_centroids[i])
   # his_r,his_k#(batch,features)
    sign=False
    for i in range(max(batch_size,max(mem.len(),reservoir.len()))):
        his_r=reservoir.sample()
        if mem.len()==0:
            continue
        else:
            sign=True
            his_k=mem.sample()
    """
    opt.zero_grad()
    his_r=reservoir.sample()
    pred_r=model(his_r.cuda().permute(0,2,1))
    #print("x,y",x,y)
    #l1=criterion(x.float(),y.float())
     #l2=criterion(his_k.cuda(),pred_k.cuda())
        #l=config.alpha*l1+0.2*l3+0.1*l2
        #l.backward(retain_graph=True)    
    #print(pred_k.device,pred_r.device,y.device,his_k.device,his_r.device,x.device)
    
    loss=criterion([pred_r,y],[his_r,x],[log_var_a,log_var_b])
    
    loss.backward()
    opt.step()
    ema.update()
    if reservoir.is_full():
            reservoir.empty()
    return model,reservoir
from train_strategy import vali
import os
import torch.nn as nn
class EarlyStopping:
    def __init__(self,config,patience=7,verbose=False,dataset_name='',delta=0.0001):
        self.delta=config.delta
        self.patience=config.patience
        self.verbose=verbose
        self.counter=0
        self.best_score=None
        self.early_stop=False
        self.val_loss_min=np.Inf
        #self.delta=delta
        self.dataset=dataset_name
    def __call__(self,val_loss,model,path):
        score=0
        score-=val_loss
        if self.best_score is None:
            self.best_score=score
            self.save_checkpoint(val_loss,model,path)
        elif score<self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter:{self.counter} out of {self.patience}')
            if self.counter>= self.patience:
                self.early_stop=True
        else:
            self.best_score=score
            self.save_checkpoint(val_loss,model,path)
            self.counter=0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(f'Validation loss decreased({self.val_loss_min:.6f}-->{val_loss:.6f}).Saving model...')
        torch.save(model.state_dict(),os.path.join(path,str(self.dataset)+'_checkpoint.pth'))
        self.val_loss_min=val_loss

def train_PR(train_loader,vali_loader,model,criterion,opt,ema,config,buffer_size=10000,batch_size=16):
    mem=buffer(buffer_size)
    early_stopping=EarlyStopping(patience=7,config=config,verbose=True,dataset_name=config.dataset)
    epochs=config.epochs
    reservoir=buffer(buffer_size) 
    lr=config.lr
    wd=0#config.wd
    log_var_a=torch.zeros((1,),requires_grad=True)
    log_var_b=torch.zeros((1,),requires_grad=True)
    log_var_c=torch.zeros((1,),requires_grad=True)
    params=([p for p in model.parameters()]+[log_var_a]+[log_var_b]+[log_var_c])
    if opt=="SGD":
        optim=torch.optim.SGD(params=params,lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(params=params,lr=lr,weight_decay=wd)
    elif opt=="AdamW":
        optim=torch.optim.AdamW(params=params,lr=lr,weight_decay=wd)
    #kmeans=MiniBatchKMeans(n_clusters=)
    model=model.cuda()
    for epoch in tqdm(range(epochs)):
        iter_count=0
        loss1_list=[]
        model.train()
        for x in tqdm(train_loader):
            x=x.float().cuda()
            columns_num=x.shape[-1]
            padding_num=0 if columns_num%2==0 else 1
            x=F.pad(x,(0,padding_num),'constant',0)
            rec_x=model(x.permute(0,2,1))
            #rec_loss=criterion(rec_x,x)
            model,mem,reservoir=train_Kmeans(model=model,mem=mem,reservoir=reservoir,x=x,y=rec_x,criterion=criterion,opt=optim,ema=ema,config=config,batch_size=batch_size,log_var_a=log_var_a,log_var_b=log_var_b,log_var_c=log_var_c)#(model,mem,reservoir,x,y,criterion,opt,config,batch_size=32)
            ema.update()
        vali_loss=vali(model,vali_loader,nn.MSELoss())
        early_stopping(vali_loss,model,path="/data/home/Linuo/iot/ILAD/wt")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch<25:
            adjust_learning_rate(optim,epoch+1,lr)
    return model

def train_ER(train_loader,vali_loader,model,criterion,opt,ema,config,buffer_size=10000,batch_size=16):
    mem=buffer(buffer_size)
    early_stopping=EarlyStopping(patience=7,config=config,verbose=True,dataset_name=config.dataset)
    epochs=config.epochs
    #reservoir=buffer(buffer_size) 
    lr=config.lr
    wd=0#config.wd
    log_var_a=torch.zeros((1,),requires_grad=True)
    log_var_b=torch.zeros((1,),requires_grad=True)
    #log_var_c=torch.zeros((1,),requires_grad=True)
    params=([p for p in model.parameters()]+[log_var_a]+[log_var_b])
    if opt=="SGD":
        optim=torch.optim.SGD(params=params,lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(params=params,lr=lr,weight_decay=wd)
    elif opt=="AdamW":
        optim=torch.optim.AdamW(params=params,lr=lr,weight_decay=wd)
    #kmeans=MiniBatchKMeans(n_clusters=)
    model=model.cuda()
    for epoch in tqdm(range(epochs)):
        iter_count=0
        loss1_list=[]
        model.train()
        for x in tqdm(train_loader):
            x=x.float().cuda()
            columns_num=x.shape[-1]
            padding_num=0 if columns_num%2==0 else 1
            x=F.pad(x,(0,padding_num),'constant',0)
            rec_x=model(x.permute(0,2,1))
            #rec_loss=criterion(rec_x,x)
            model,mem=train_ER_(model=model,reservoir=mem,x=x,y=rec_x,criterion=criterion,opt=optim,ema=ema,config=config,batch_size=batch_size,log_var_a=log_var_a,log_var_b=log_var_b)#(model,mem,reservoir,x,y,criterion,opt,config,batch_size=32)
            ema.update()
        vali_loss=vali(model,vali_loader,nn.MSELoss())
        early_stopping(vali_loss,model,path="/data/home/Linuo/iot/ILAD/wt")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch<25:
            adjust_learning_rate(optim,epoch+1,lr)
    return model