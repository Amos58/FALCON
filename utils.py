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
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calculate_wcss(X, labels, centroids):
    wcss=[]
    for i in range(len(labels)):
        wcss.append((X[i]-centroids[labels[i]])**2)
    return np.sum(np.array(wcss))

from statsmodels.tsa.seasonal import STL
import numpy as np
from sklearn.preprocessing import StandardScaler

def PSM_data(tensor=True):
    train_path="/data/home/Linuo/iot/dataset/unsupervised_AD/PSM/train.csv"
    test_path="/data/home/Linuo/iot/dataset/unsupervised_AD/PSM/test.csv"
    labels_path="/data/home/Linuo/iot/dataset/unsupervised_AD/PSM/test_label.csv"
    train_df=pd.read_csv(train_path,dtype=np.float32).fillna(0)
    test_df=pd.read_csv(test_path,dtype=np.float32).fillna(0)
    train_labels_df=pd.read_csv(labels_path,dtype=np.float32).fillna(0)
    train,test,labels=train_df.drop("timestamp_(min)",axis=1).values,test_df.drop("timestamp_(min)",axis=1).values,train_labels_df.drop("timestamp_(min)",axis=1).values
    scaler=StandardScaler()
    scaler.fit(train)
    test=scaler.transform(test)
    train=scaler.transform(train)
    if tensor:
        return torch.tensor(train),torch.tensor(test),torch.tensor(labels)
    return train,test,labels

def anoshift_data(id="IID",tensor=True):
    train_path="/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/train.npy"
    NEAR_test_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2011.npy","/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2012.npy", "/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2013.npy"]
    FAR_test_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2014.npy","/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2015.npy"]
    NEAR_labels_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2011.npy","/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2012.npy", "/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2013.npy"]
    FAR_labels_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2014.npy","/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2015.npy"]
    #FAR_test_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2014.npy","/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2015.npy"]
    IID_test_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2006.npy","/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2007.npy", "/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2008.npy", "/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2009.npy", "/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testX_2010.npy"]
    IID_labels_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2006.npy","/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2007.npy", "/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2008.npy", "/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2009.npy", "/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2010.npy"]
    #IID_test_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2014.npy","/data/home/Linuo/iot/dataset/unsupervised_AD/datasets/Kyoto-2016_AnoShift/testY_2015.npy"]
    train=np.load(train_path).astype(np.float32)
    if id=="IID":
        IID_len=len(IID_test_path)
        test_=[]
        labels_=[]
        for i in range(IID_len):
            testt=np.load(IID_test_path[i])
            labell=np.load(IID_labels_path[i])
            testt=testt.astype(np.float32)
            labell=labell.astype(np.float32)
            test_.append(testt)
            labels_.append(labell)
            test=np.concatenate([i for i in test_])
            labels=np.concatenate([i for i in labels_])
            scaler=StandardScaler()
            scaler.fit(train)
            test=scaler.transform(test)
            train=scaler.transform(train)
    elif id=="NEAR":
        NEAR_len=len(NEAR_test_path)
        test_=[]
        labels_=[]
        for i in range(NEAR_len):
            testt=np.load(NEAR_test_path[i])
            labell=np.load(NEAR_labels_path[i])
            testt=testt.astype(np.float32)
            labell=labell.astype(np.float32)
            test_.append(testt)
            labels_.append(labell)
        test=np.concatenate([i for i in test_])
        labels=np.concatenate([i for i in labels_])
        scaler=StandardScaler()
        scaler.fit(train)
        test=scaler.transform(test)
        train=scaler.transform(train)
    elif id=="FAR":
        FAR_len=len(FAR_test_path)
        test_=[]
        labels_=[]
        for i in range(FAR_len):
            testt=np.load(FAR_test_path[i])
            labell=np.load(FAR_labels_path[i])
            testt=testt.astype(np.float32)
            labell=labell.astype(np.float32)
            test_.append(testt)
            labels_.append(labell)
        test=np.concatenate((test_[0],test_[1]))
        labels=np.concatenate((labels_[0],labels_[1]))      
        scaler=StandardScaler()
        scaler.fit(train)
        test=scaler.transform(test)
        train=scaler.transform(train)
    if tensor:
        return torch.tensor(train),torch.tensor(test),torch.tensor(labels)
    return train,test,labels

class Anoshift(Dataset):
    def __init__(self,X,Y=None,seq_len=100,decom=False,period=100,stride=50,freq=False,ac=False):
        self.decom=decom
        split=[]
        #print(Y)
        self.Y=Y
        if self.Y is not None:
            split_y=[]
            #print("test")
            for i in range(1,len(Y)-seq_len,1):
                split_y.append(int(any(Y[i:i+seq_len])))
            self.Y=split_y
        #split
        for i in range(1,len(X)-seq_len,1):
            split.append(X[i:i+seq_len])
        if decom:
            self.trend=[]
            self.season=[]
            self.res=[]
            for i in split:
                decomposition=STL(i,period=period).fit()
                self.trend.append(torch.LongTensor(decomposition.trend))
                self.season.append(torch.LongTensor(decomposition.seasonal))
                self.res.append(torch.LongTensor(decomposition.resid))
        #if freq:

        #print(len(split[0]))
        self.X=torch.stack(split)

    def __getitem__(self,idx):
        if self.Y is not None:    
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx],self.Y[idx]
            else:
                return self.X[idx],self.Y[idx]
        else:
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx]
            else:
                return self.X[idx]
    def __len__(self):
        return len(self.X)
    
class PSM(Dataset):
    def __init__(self,X,Y=None,seq_len=100,decom=False,period=100,stride=50,freq=False,ac=False):
        self.decom=decom
        split=[]
        #print(Y)
        self.Y=Y
        if self.Y is not None:
            split_y=[]
            #print("test")
            for i in range(1,len(Y)-seq_len,1):
                split_y.append(int(any(Y[i:i+seq_len])))
            self.Y=split_y
        #split
        for i in range(1,len(X)-seq_len,1):
            split.append(X[i:i+seq_len])
        if decom:
            self.trend=[]
            self.season=[]
            self.res=[]
            for i in split:
                decomposition=STL(i,period=period).fit()
                self.trend.append(torch.LongTensor(decomposition.trend))
                self.season.append(torch.LongTensor(decomposition.seasonal))
                self.res.append(torch.LongTensor(decomposition.resid))
        #if freq:

        #print(len(split[0]))
        self.X=torch.stack(split)

    def __getitem__(self,idx):
        if self.Y is not None:    
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx],self.Y[idx]
            else:
                return self.X[idx],self.Y[idx]
        else:
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx]
            else:
                return self.X[idx]
    def __len__(self):
        return len(self.X)
    
def tesla_data(tensor=True):
    train_path="/data/home/Linuo/iot/dataset/unsupervised_AD/tesla/normal.csv"
    test_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/tesla/3C2#Spoof.csv","/data/home/Linuo/iot/dataset/unsupervised_AD/tesla/3F5#Spoof.csv","/data/home/Linuo/iot/dataset/unsupervised_AD/tesla/273#Spoof.csv","/data/home/Linuo/iot/dataset/unsupervised_AD/tesla/DoS.csv","/data/home/Linuo/iot/dataset/unsupervised_AD/tesla/fuzzing.csv"]
    #labels_path="/data/home/Linuo/iot/dataset/unsupervised_AD/PSM/test_label.csv"
    train_df=pd.read_csv(train_path,dtype=np.float32).fillna(0)
    spoof1,spoof2,spoof3,dos,fuzzy=pd.read_csv(test_path[0],dtype=np.float32),pd.read_csv(test_path[1],dtype=np.float32),pd.read_csv(test_path[2],dtype=np.float32),pd.read_csv(test_path[3],dtype=np.float32),pd.read_csv(test_path[4],dtype=np.float32)
    labels1,labels2,labels3,labels4,label5=spoof1["name"],spoof2["name"],spoof3["name"],dos["name"],fuzzy["name"]
    #test_df=pd.read_csv(test_path,dtype=np.float32).fillna(0)
    #train_labels_df=pd.read_csv(labels_path,dtype=np.float32).fillna(0)
    train,spoof1,spoof2,spoof3,dos,fuzzy=train_df.drop("index",axis=1).drop("name",axis=1).drop("timestamp",axis=1).values,spoof1.drop("name",axis=1).drop("index",axis=1).drop("timestamp",axis=1).values,spoof2.drop("name",axis=1).drop("index",axis=1).drop("timestamp",axis=1).values,spoof3.drop("name",axis=1).drop("index",axis=1).drop("timestamp",axis=1).values,dos.drop("name",axis=1).drop("index",axis=1).drop("timestamp",axis=1).values,fuzzy.drop("name",axis=1).drop("index",axis=1).drop("timestamp",axis=1).values
    scaler=StandardScaler()
    scaler.fit(train)
    spoof1,spoof2,spoof3,dos,fuzzy=scaler.transform(spoof1),scaler.transform(spoof2),scaler.transform(spoof3),scaler.transform(dos),scaler.transform(fuzzy)
    train=scaler.transform(train)
    if tensor:
        return torch.tensor(train),torch.tensor(spoof1),torch.tensor(labels1),torch.tensor(spoof2),torch.tensor(labels2),torch.tensor(spoof3),torch.tensor(labels3),torch.tensor(dos),torch.tensor(labels4),torch.tensor(fuzzy),torch.tensor(label5)
    return train,spoof1,labels1,spoof2,labels2,spoof3,labels3,dos,labels4,fuzzy,label5

class Tesla(Dataset):
    def __init__(self,X,Y=None,seq_len=100,decom=False,period=100,stride=50,freq=False,ac=False):
        self.decom=decom
        split=[]
        #print(Y)
        self.Y=Y
        if self.Y is not None:
            split_y=[]
            #print("test")
            for i in range(1,len(Y)-seq_len,1):
                split_y.append(int(any(Y[i:i+seq_len])))
            self.Y=split_y
        #split
        for i in range(1,len(X)-seq_len,1):
            split.append(X[i:i+seq_len])
        if decom:
            self.trend=[]
            self.season=[]
            self.res=[]
            for i in split:
                decomposition=STL(i,period=period).fit()
                self.trend.append(torch.LongTensor(decomposition.trend))
                self.season.append(torch.LongTensor(decomposition.seasonal))
                self.res.append(torch.LongTensor(decomposition.resid))
        #if freq:

        #print(len(split[0]))
        self.X=torch.stack(split)

    def __getitem__(self,idx):
        if self.Y is not None:    
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx],self.Y[idx]
            else:
                return self.X[idx],self.Y[idx]
        else:
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx]
            else:
                return self.X[idx]
    def __len__(self):
        return len(self.X)
    

def SMD_data(id,tensor=True):
    train_path=["/data/home/Linuo/iot/ServerMachineDataset/train/train_1.txt","/data/home/Linuo/iot/ServerMachineDataset/train/train_2.txt","/data/home/Linuo/iot/ServerMachineDataset/train/train_3.txt"]
    test_path=["/data/home/Linuo/iot/ServerMachineDataset/test/test_1.txt","/data/home/Linuo/iot/ServerMachineDataset/test/test_2.txt","/data/home/Linuo/iot/ServerMachineDataset/test/test_3.txt"]
    label_path=["/data/home/Linuo/iot/ServerMachineDataset/test_label/test_label_1.txt","/data/home/Linuo/iot/ServerMachineDataset/test_label/test_label_2.txt","/data/home/Linuo/iot/ServerMachineDataset/test_label/test_label_3.txt"]
    train_data,test_data,labels=np.loadtxt(train_path[id-1],dtype=np.float32,delimiter=","),np.loadtxt(test_path[id-1],dtype=np.float32,delimiter=","),np.loadtxt(label_path[id-1],dtype=np.float32,delimiter=",")
    scaler=StandardScaler()
    scaler.fit(train_data)
    train_data=scaler.transform(train_data)
    test_data=scaler.transform(test_data)
    if tensor:
        return torch.tensor(train_data),torch.tensor(test_data),np.loadtxt(label_path[id-1],dtype=np.float32,delimiter=",")
    else:
        return train_data,test_data,np.loadtxt(label_path[id-1],dtype=np.float32,delimiter=",")
def SMD_data_split(id,tensor=True):
    train_path=["/data/home/Linuo/iot/ServerMachineDataset/train/machine-1-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-1-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-1-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-1-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-1-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-1-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-1-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-1-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-2-9.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-9.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-10.txt",
"/data/home/Linuo/iot/ServerMachineDataset/train/machine-3-11.txt"]
    test_path=["/data/home/Linuo/iot/ServerMachineDataset/test/machine-1-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-1-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-1-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-1-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-1-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-1-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-1-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-1-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-2-9.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-9.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-10.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test/machine-3-11.txt"]
    label_path=["/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-1-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-1-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-1-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-1-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-1-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-1-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-1-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-1-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-2-9.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-1.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-2.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-3.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-4.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-5.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-6.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-7.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-8.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-9.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-10.txt",
"/data/home/Linuo/iot/ServerMachineDataset/test_label/machine-3-11.txt"]



    train_data,test_data,labels=np.loadtxt(train_path[id-1],dtype=np.float32,delimiter=","),np.loadtxt(test_path[id-1],dtype=np.float32,delimiter=","),np.loadtxt(label_path[id-1],dtype=np.float32,delimiter=",")
    scaler=StandardScaler()
    scaler.fit(train_data)
    train_data=scaler.transform(train_data)
    test_data=scaler.transform(test_data)
    if tensor:
        return torch.tensor(train_data),torch.tensor(test_data),np.loadtxt(label_path[id-1],dtype=np.float32,delimiter=",")
    else:
        return train_data,test_data,np.loadtxt(label_path[id-1],dtype=np.float32,delimiter=",")
    




class SMD(Dataset): 
    def __init__(self,X,Y=None,seq_len=100,decom=False,period=100,stride=50,freq=False,ac=False):
        self.decom=decom
        split=[]
        #print(Y)
        self.Y=Y
        if self.Y is not None:
            split_y=[]
            #print("test")
            for i in range(1,len(Y)-seq_len,1):
                split_y.append(int(any(Y[i:i+seq_len])))
            #print(split_y[0])
            #print(len(split_y))
            #print(type(split_y[0]))
            #print(type(split_y))
            
            self.Y=split_y
        #split
        for i in range(1,len(X)-seq_len,1):
            split.append(X[i:i+seq_len])
        if decom:
            self.trend=[]
            self.season=[]
            self.res=[]
            for i in split:
                decomposition=STL(i,period=period).fit()
                self.trend.append(torch.LongTensor(decomposition.trend))
                self.season.append(torch.LongTensor(decomposition.seasonal))
                self.res.append(torch.LongTensor(decomposition.resid))
        #if freq:

        #print(len(split[0]))
        self.X=torch.stack(split)

        #print(len(self.X[0]))
    def __getitem__(self,idx):
        if self.Y is not None:    
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx],self.Y[idx]
            else:
                return self.X[idx],self.Y[idx]
        else:
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx]
            else:
                return self.X[idx]
    def __len__(self):
        return len(self.X)
    

import pandas as pd

def MSL_SMAP_data(dataset="MSL",tensor=True):
    train_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/SMAP/processed_csv/MSL_train.csv","/data/home/Linuo/iot/dataset/unsupervised_AD/SMAP/processed_csv/SMAP_train.csv"]
    test_path=["/data/home/Linuo/iot/dataset/unsupervised_AD/SMAP/processed_csv/MSL_test.csv","/data/home/Linuo/iot/dataset/unsupervised_AD/SMAP/processed_csv/SMAP_test.csv"]
    
    if dataset=="MSL":
        train_df,test_df=pd.read_csv(train_path[0],dtype=np.float32),pd.read_csv(test_path[0],dtype=np.float32)
        train_raw,test_raw,labels=train_df.drop("timestamp",axis=1).values,test_df.drop("timestamp",axis=1).drop("label",axis=1).values,test_df["label"].values
    elif dataset=="SMAP":
        train_df,test_df=pd.read_csv(train_path[1],dtype=np.float32),pd.read_csv(test_path[1],dtype=np.float32)
        train_raw,test_raw,labels=train_df.drop("timestamp",axis=1).values,test_df.drop("timestamp",axis=1).drop("label",axis=1).values,test_df["label"].values
    scaler=StandardScaler()
    scaler.fit(train_raw)
    train_raw=scaler.transform(train_raw)
    test_raw=scaler.transform(test_raw)
    if tensor:
        return torch.tensor(train_raw),torch.tensor(test_raw),torch.tensor(labels)
    else:
        return train_raw,test_raw,labels

class MSL_SMAP(Dataset):
    def __init__(self,X,Y=None,seq_len=100,decom=False,period=100,stride=50,freq=False,ac=False):
        self.decom=decom
        split=[]
        #print(Y)
        self.Y=Y
        if self.Y is not None:
            split_y=[]
            #print("test")
            for i in range(1,len(Y)-seq_len,1):
                split_y.append(int(any(Y[i:i+seq_len])))
            self.Y=split_y
        #split
        for i in range(1,len(X)-seq_len,1):
            split.append(X[i:i+seq_len])
        if decom:
            self.trend=[]
            self.season=[]
            self.res=[]
            for i in split:
                decomposition=STL(i,period=period).fit()
                self.trend.append(torch.LongTensor(decomposition.trend))
                self.season.append(torch.LongTensor(decomposition.seasonal))
                self.res.append(torch.LongTensor(decomposition.resid))
        #if freq:

        #print(len(split[0]))
        self.X=torch.stack(split)

    def __getitem__(self,idx):
        if self.Y is not None:    
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx],self.Y[idx]
            else:
                return self.X[idx],self.Y[idx]
        else:
            if self.decom:
                return self.trend[idx],self.season[idx],self.res[idx]
            else:
                return self.X[idx]
    def __len__(self):
        return len(self.X)
    


def threshold_update(threshold,pred,y,interval=1e-4):
    if pred==0 and y==1:
        threshold=threshold-interval*3
    elif pred==1 and y==0:
        threshold=threshold+interval*3
    else:
        threshold=threshold+interval
    return threshold


def anomaly(preds,threshold=0.1):
    if preds<=threshold:
        return 0
    else:
        return 1
    
def plot_loss(loss,figure_size):
    hight,width=figure_size
    plt.figure(figsize=(hight,width))
    
    sns.lineplot(np.array(loss))
    plt.show()


def train_without_CL(criterion,optim,train_loader,model,device,epoch):
    criterion=criterion.to(device)
    optim=optim
    model=model.to(device)
    L=[]
    for i in tqdm(range(epoch)):
        l=[] 
        for x in train_loader:
            optim.zero_grad
            x=x.cuda()
            pred=model(x)
            loss=criterion(pred,x)
            l.append(loss.detach().cpu().item())
            loss.backward()
            optim.step()
        L.append(np.mean(l))
    return model,L
class EarlyStopping:
    def __init__(self,patience=7,verbose=False,dataset_name='',delta=0.0001):
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

def vali(model,vali_loader,criterion):
    model.eval()
    loss_1=[]
    for input_data in tqdm(vali_loader):
        input=input_data.float().cuda()
        columns_num=input.shape[-1]
        padding_num=0 if columns_num%2==0 else 1
        input=F.pad(input,(0,padding_num),'constant',0)
        output=model(input.permute(0,2,1))
        rec_loss=criterion(output,input)
        loss_1.append(rec_loss.item())
    return np.average(loss_1)
def adjust_learning_rate(opt,epoch,lr_):
    lr_adjust={epoch:lr_*(0.95**((epoch-1)//1))}
    if epoch in lr_adjust.keys():
        lr=lr_adjust[epoch]
        for param_group in opt.param_groups:
            param_group['lr']=lr
        print('Updating learning rate to {}'.format(lr))
def train(model,train_loader,vali_loader,config,opt,criterion):
    early_stopping=EarlyStopping(patience=7,verbose=True,dataset_name="SMD")
    epochs=config.epochs
   # vali_loader=train_loader[:int(len(train_loader)*0.8)]
    lr=config.lr
    wd=config.wd
    if opt=="SGD":
        optim=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="AdamW":
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    if criterion=="MSE":
        criterions=torch.nn.MSELoss()
    Loss=[]
    for epoch in tqdm(range(epochs)):
        iter_count=0
        loss1_list=[]
        model.train()
        for i,xx in enumerate(train_loader):
            optim.zero_grad()
            iter_count+=1
            columns_num=xx.shape[-1]
            padding_num=0 if columns_num%2==0 else 1
            xx=F.pad(xx,(0,padding_num),'constant',0)
            #print(xx.shape)
            #print(xx.shape)
            #x=xx.float().cuda()[:,::config.DSR,:]
            #print(x.shape)
            #print(xx.shape)
            x_rec=model(xx.permute(0,2,1).cuda())#(100,38)
            #print(x_rec.shape)
            #break
            #print(x_rec.shape)
            #print("x_rec",x_rec.shape,xx.shape)
            rec_loss=criterions(x_rec,xx.cuda())
            loss1_list.append(rec_loss.item())
            loss1=rec_loss
            loss1.backward(retain_graph=True)
            optim.step()
            ema.update()
        #break
        train_loss=np.average(loss1_list)
        print("train_loss",train_loss)
        vali_loss=vali(model,vali_loader,criterions)
        early_stopping(vali_loss,model,path="/data/home/Linuo/iot/ILAD/wt")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch<25:
            adjust_learning_rate(optim,epoch+1,lr)
    return model,Loss,optim


def test(test_loader,model,label,criterion):
    results=[[] for i in range(300000)]
    cnt=0
    for x in tqdm(test_loader):
        x=x.cuda()
        rec=model(x)
        tmp_x=x.detach().cpu().view(-1,38)
        tmp_rec=rec.detach().cpu().view(-1,38)
        for i in range(0,tmp_rec.shape[0]-1):
            results[cnt+i].append(criterion(tmp_rec[i],tmp_x[i]).item())
        cnt+=1
    ans=[]
    for i in results:
        ans.append(np.mean(i))
    ans=np.array(ans)
    np.nan_to_num(ans,0)
    ans_1=ans[:len(label)]
    #ts_metrics(labels,point_adjustment(labels,ans_1))
    return ts_metrics(label,point_adjustment(label,ans_1))


def test_with_navie_tta(test_loader,model,label,criterion,opt,threshold):
    results=[[] for i in range(194374)]
    cnt=0
    for x in tqdm(test_loader):
        x=x.cuda()
        #cnt+=1
        y=model(x)
        if criterion(y,x).detach().cpu().item()<threshold:
            opt.zero_grad()
            loss=criterion(y,x)
            loss.backward()
            opt.step()
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

def train_ER(model,his,x,y,criterion,opt):
    his_y=model(his.cuda())
    l1=criterion(his_y,his)
    l2=criterion(y,x)
    l=l1+l2
    opt.zero_grad
    l.backward()
    opt.step()
    return model



def test_with_er_tta(test_loader,model,label,criterion,opt,threshold,buffer_size=10000):
    mem=buffer(buffer_size)
    results=[[] for i in range(len(test_loader))]
    for x in tqdm(test_loader):
        x=x.cuda()
        y=model(x)
        l=criterion(y,x)
        if l.detach().cpu().item() <threshold:
            mem.update(x.detach().cpu())
            his=mem.sample()
            model=train_ER(model,his,x,criterion,opt)
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


def train_Kmeans(model,mem,reservoir,x,y,criterion,opt,batch_size=32):
    reservoir.update(x)
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
    opt.zero_grad
    his_r=torch.stack(his_r).view(batch_size,100,38)
    #print(type(his_r))
    #print(his_r.shape)
    #print(type(his_r[0]))
    #print(his_r[0])
    pred_r=model(his_r)
    l1=criterion(x,y)
    l3=criterion(his_r,pred_r)
    if sign:
        pred_k=model(his_k)
        l2=criterion(his_k,pred_k)
        l=l1+l3+l2
        l.backward()    
    else:
        l=l1+l3
        l.backward()
    opt.step()
    if reservoir.is_full():
            reservoir.empty()
    return model,mem,reservoir

def test_with_kmeans_tta(test_loader,model,label,criterion,opt,threshold,features_nums,buffer_size=10000,batch_size=32):
    mem=buffer(buffer_size)
    reservoir=buffer(buffer_size) 
    cnt=0
    results=[[] for i in range(len(test_loader)+1)]
    #kmeans=MiniBatchKMeans(n_clusters=)
    model=model.cuda()
    for x,_ in tqdm(test_loader):
        x=x.cuda()
        y=model(x)
        l=criterion(y,x)
        if l.detach().cpu().item() <threshold:
            model,mem,reservoir=train_Kmeans(model,mem,reservoir,x,y,criterion,opt,batch_size)
        tmp_x=x.detach().cpu().view(-1,features_nums)
        tmp_y=y.detach().cpu().view(-1,features_nums)
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

from tqdm import tqdm
def train(model,train_loader,config,opt,criterion):
    epochs=config["epoch"]
    lr=config["lr"]
    wd=config["weight_decay"]
    if opt=="SGD":
        optim=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    if criterion=="MSE":
        criterions=torch.nn.MSELoss()
    Loss=[]
    for epoch in tqdm(range(epochs)):
        for x in train_loader:
            x=x.cuda()
            x_rec,_=model(x)
            optim.zero_grad()
            loss=criterions(x_rec,x)
            loss.backward()
            optim.step()
        Loss.append(loss.detach().cpu().item().mean())
        print(Loss[-1])
    return model,Loss,optim 
import torch.nn as nn

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            param.requires_grad=True
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            param.requires_grad=True
            if param.requires_grad:
                #assert name in self.shadow
                #print("111")
                #print(name)
                #print(param)
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            param.requires_grad=True
            if param.requires_grad:
                #assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            param.requires_grad=True
            if param.requires_grad:
                #assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    def values(self):
        for name, param in self.model.named_parameters():
            print(name)



def test_with_navie_tta(test_loader,model,label,criterion,opt,threshold,ema):
    ema.apply_shadow()
    #results=[[] for i in range(194374)]
    attens_energy=[]
    test_labels=[]
    cnt=0
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
                opt.zero_grad()
                loss=criterion(x,x_rec)
                loss.backward()
                opt.step()
    #ts_metrics(labels,point_adjustment(labels,ans_1))
    return ts_metrics(test_labels,point_adjustment(test_labels,attens_energy))

def train(model,train_loader,vali_loader,config,opt,criterion,ema):
    early_stopping=EarlyStopping(patience=7,verbose=True,dataset_name=config.dataset)
    epochs=config.epochs
   # vali_loader=train_loader[:int(len(train_loader)*0.8)]
    lr=config.lr
    wd=config.wd
    if opt=="SGD":
        optim=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="Adam":
        optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    elif opt=="AdamW":
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    if criterion=="MSE":
        criterions=torch.nn.MSELoss()
    Loss=[]
    for epoch in tqdm(range(epochs)):
        iter_count=0
        loss1_list=[]
        model.train()
        for i,xx in enumerate(train_loader):
            optim.zero_grad()
            iter_count+=1
            columns_num=xx.shape[-1]
            padding_num=0 if columns_num%2==0 else 1
            xx=F.pad(xx,(0,padding_num),'constant',0)
            #print(xx.shape)
            #print(xx.shape)
            #x=xx.float().cuda()[:,::config.DSR,:]
            #print(x.shape)
            #print(xx.shape)
            x_rec=model(xx.permute(0,2,1).cuda())#(100,38)
            #print(x_rec.shape)
            #break
            #print(x_rec.shape)
            #print("x_rec",x_rec.shape,xx.shape)
            rec_loss=criterions(x_rec,xx.cuda())
            loss1_list.append(rec_loss.item())
            loss1=rec_loss
            loss1.backward(retain_graph=True)
            optim.step()
            ema.update()
        #break
        train_loss=np.average(loss1_list)
        print("train_loss",train_loss)
        vali_loss=vali(model,vali_loader,criterions)
        early_stopping(vali_loss,model,path="/data/home/Linuo/iot/ILAD/wt")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch<25:
            adjust_learning_rate(optim,epoch+1,lr)
    return model,Loss,optim