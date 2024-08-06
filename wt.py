from torch.utils.data import DataLoader
from utils import SMD,SMD_data,same_seeds,MSL_SMAP_data,MSL_SMAP,PSM,PSM_data,tesla_data,Tesla
from utils import EMA
from deepod.metrics import ts_metrics
from pytorch_wavelets import DWTForward
same_seeds(42)
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import os
from deepod.metrics import ts_metrics,point_adjustment
from tqdm import tqdm
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from model import Dlinear
# SMTP 服务器配置
smtp_server = 'smtp.zju.edu.cn'
smtp_port = 25  # 端口号，对于SMTP通常为25或587

# 邮箱账户和密码
email_user = '22321057@zju.edu.cn' # 替换为你的ZJU邮箱账号
email_pass = '123456Qwe'  # 替换为你的邮箱密码

# 发件人和收件人
sender = email_user
receivers = ['amos_ln@foxmail.com']  # 替换为QQ邮箱地址


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

import torch.nn as nn
import pytorch_wavelets as wavelets

        
class TFTS(nn.Module):
    def __init__(self,config):
        super(TFTS,self).__init__()
        self.win_size=config.win_size
        self.seq_len=config.win_size//4
        self.pred_len=config.win_size-config.win_size//4
        #kernel_size=25
        self.dominance_freq=config.cutfreq
        #self.alpha=torch.ParameterDict()
        self.length_ratio=(self.seq_len+self.pred_len)/self.pred_len
        self.freq_upsampler=nn.Linear(self.dominance_freq,int(self.dominance_freq*self.length_ratio)).to(torch.cfloat)
        self.kernel_size=config.kernel_size
        self.decompose=series_decomp(self.kernel_size)
        self.Linear_Seasonal = nn.Linear(self.win_size,self.win_size)
        self.Linear_Trend = nn.Linear(self.win_size,self.win_size)
        self.Linear_Decoder = nn.Linear(self.win_size,self.win_size)
        self.Linear_Seasonal.weight = nn.Parameter((1/self.win_size)*torch.ones([self.win_size,self.win_size]))
        self.Linear_Trend.weight = nn.Parameter((1/self.win_size)*torch.ones([self.win_size,self.win_size]))
    def dlinear(self,x):
        #print(x.shape)
        seasonal_init, trend_init = self.decompose(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        #print(len(seasonal_init[0][0]))
        #print(self.Linear_Seasonal)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        #print("Dlinear",trend_output.shape)
        return (seasonal_output+trend_output).permute(0,2,1)

    def forward(self,x,tx):
        # RIN
        x_mean=torch.mean(x,dim=1,keepdim=True)
        x=x-x_mean
        x_var=torch.var(x,dim=1,keepdim=True)+1e-5
        x=x/torch.sqrt(x_var)
        #fft
        low_specx=torch.fft.rfft(x,dim=1)
        #高通滤波？
        low_specx[:,self.dominance_freq:]=0         
        low_specx=low_specx[:,0:self.dominance_freq,:]
        #上采样
        low_specx_=self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        low_specxy=torch.zeros([low_specx_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specx_.size(2)],dtype=low_specx_.dtype).to(low_specx_.device)
        low_specxy[:,0:low_specx_.size(1),:]=low_specx_

        low_xy=torch.fft.irfft(low_specxy,dim=1)
        low_xy=low_xy*self.length_ratio
        xy=(low_xy)*torch.sqrt(x_var)+x_mean
        rec_x=self.dlinear(tx)
        #print(rec_x.shape)
        #print(xy.shape)
        return rec_x

import seaborn as sns
import pytorch_wavelets as wavelets
import torch.nn as nn
import numpy as np
class LinearBlock(nn.Module):
    def __init__(self,win_size):
        super(LinearBlock,self).__init__()
        self.seq_len=win_size
        #self.layer=config.layer
        #self.Linear=nn.ModuleList()
        #self.block=#nn.Sequential(
        self.block=nn.Linear(self.seq_len//2,self.seq_len//2)
        nn.init.xavier_uniform_(self.block.weight)

            #nn.LeakyReLU(),
            #nn.Dropout(config.dropout)
            #)
    def forward(self,x):
        return self.block(x)
    
class LinearLayer(nn.Module):
    def __init__(self,layer,win_size):
        super(LinearLayer,self).__init__()
        self.layer=layer
        self.block=LinearBlock(win_size)
    def forward(self,x):
        for i in range(self.layer):
            x=self.block(x)
        return x

class HMLP(nn.Module):
    def __init__(self,config):
        super(HMLP,self).__init__()
        self.win_size=config.win_size
        self.layer=config.layer
        self.batch_size=config.batch_size
        self.nets=nn.ModuleList()
        #self.LinearBlock=nn.Sequential(nn.Linear(self.win_size//2,self.win_size//2),
        #                       nn.BatchNorm2d(32),
        #                       nn.ReLU(),
        #                       nn.Dropout(0.2))
        self.decompose_layer=3
        for i in range(self.decompose_layer+1):
            self.nets.append(
                LinearLayer(self.layer,self.win_size)

            )
        
    def model(self,coefs):
        new_coefs=[]
        for coef,net in zip(coefs,self.nets):
            new_coef=net(coef.permute(0,2,1))
            new_coefs.append(new_coef.permute(0,2,1))
        return new_coefs
    
    def forward(self,x):
        # RIN
        #print("x",x.shape)

        x_mean=torch.mean(x,dim=1,keepdim=True)
        x=x-x_mean
        x_var=torch.var(x,dim=1,keepdim=True)+1e-5
        x=x/torch.sqrt(x_var)

        batch,channel,length=x.shape
        #print("b,c,l",batch,channel,length)
        x=x.unsqueeze(0)
        #print(x[0][0].dtype)
        wavelet = wavelets.DWTForward(J=1, mode='zero', wave='haar').to(x.device)  # 使用haar小波
        idwt=wavelets.IDWT2D(mode='zero',wave='haar').to(x.device)
        yl,yhs=wavelet(x.permute(0,1,3,2))
        yhs_s=torch.stack(yhs).view(3,batch,length//2,int(channel/2+0.5))
        coefs=torch.concat([yl,yhs_s],dim=0)#.permute(2,1,3,4)
        coefs_new=self.model(coefs)
        yhs_s=torch.stack(coefs_new[1:]).view(1,1,batch,3, length//2, int(channel/2+0.5))
        out=idwt((coefs_new[0].unsqueeze(0),list(yhs_s))) #(1, 32, 19, 50),(1, 1, 32, 3, 19, 50)
        #print("out,x_var,x_mean",out.shape,x_var.shape,x_mean.shape)

        out=out.squeeze(0).permute(0,2,1)*torch.sqrt(x_var)+x_mean
        #print(out.shape)
        #rec_out=out
        #print("rec out",rec_out.shape)
        return out.permute(0,2,1)

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
        torch.save(model.state_dict(),os.path.join(path,os.path.join(str(config.model),str(self.dataset))+'_checkpoint.pth'))
        self.val_loss_min=val_loss

def vali(model,vali_loader,criterion):
    model.eval()
    loss_1=[]
    for input_data in tqdm(vali_loader):
        input_=input_data.float().cuda()
        columns_num=input_.shape[-1]
        padding_num=0 if columns_num%2==0 else 1
        input_=F.pad(input_,(0,padding_num),'constant',0)
        if config.model=="Dlinear":
            output=model(input_.permute(0,1,2).cuda())
        else:        
            output=model(input_.permute(0,2,1).cuda())
        #output=model(input.permute(0,2,1))
        rec_loss=criterion(output,input_)
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
    early_stopping=EarlyStopping(patience=7,verbose=True,dataset_name=config.dataset)#,path="/data/home/Linuo/iot/ILAD/wt/checkpoint")
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
            if config.model=="Dlinear":
                x_rec=model(xx.permute(0,1,2).cuda())
            else:        
                x_rec=model(xx.permute(0,2,1).cuda())
            #print(x_rec.shape,xx.shape)
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
        #print("train_loss",train_loss)
        vali_loss=vali(model,vali_loader,criterions)
        early_stopping(vali_loss,model,path="/data/home/Linuo/iot/ILAD/wt")
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch<25:
            adjust_learning_rate(optim,epoch+1,lr)
    return model,Loss,optim


def test(model,test_loader,config,criterion):
    ema.apply_shadow()
    model.eval()
    attens_energy=[]
    test_labels=[]
    REC=[]
    Duration=0
    path1="/data/home/Linuo/iot/ILAD/log/attens_energy"
    path2="/data/home/Linuo/iot/ILAD/log/label"
    path3="/data/home/Linuo/iot/ILAD/log/recdata"
    st=time.time()
    for (xx,y) in tqdm(test_loader):
        columns_num=xx.shape[-1]
        padding_num=0 if columns_num%2==0 else 1
        xx=F.pad(xx,(0,padding_num),'constant',0)     
        if config.model=="Dlinear":
            x_rec=model(xx.permute(0,1,2).cuda())
        else:        
            x_rec=model(xx.permute(0,2,1).cuda())
        #print("test rec",x_rec.shape)
        #rec_loss=criterion(xx[:,-1:,:],x_rec[:,-1:,:]).squeeze(1)
        #rec_loss=rec_loss.detach().cpu().numpy()
        #attens_energy.append(rec_loss)
        
        for u in range(x_rec.shape[0]):
            rec_loss=criterion(x_rec[u],xx[u].float().cuda())
            REC.append(x_rec[u].detach().cpu().numpy())
            cri=rec_loss.unsqueeze(0)
            cri=cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(y[u].unsqueeze(0))
        
    #Duration+=time.time()-st
    attens_energy=np.concatenate(attens_energy,axis=0).reshape(-1)
    test_labels=np.concatenate(test_labels,axis=0).reshape(-1)
    attens_energy=np.array(attens_energy)
    test_labels=np.array(test_labels)
    #st=time.time()
    scores=ts_metrics(test_labels,point_adjustment(test_labels,attens_energy))
    REC=np.concatenate(REC,axis=0).reshape(-1)
    ema.restore()
    np.save(os.path.join(os.path.join(path1, config.model)) + "_" + str(config.dataset) + "_" + str(config.win_size), attens_energy)
    np.save(os.path.join(os.path.join(path2, config.model)) + "_" + str(config.dataset) + "_" + str(config.win_size), test_labels)
    np.save(os.path.join(os.path.join(path3, config.model)) + "_" + str(config.dataset) + "_" + str(config.win_size), REC)


    return scores,attens_energy,test_labels,Duration

print("start")
parser=argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--win_size', type=int, default=100)
parser.add_argument('--enc_in', type=int, default=38)
parser.add_argument('--output_c', type=int, default=38)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--pretrained_model', type=str, default=None)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
parser.add_argument('--model_save_path', type=str, default='checkpoints')
parser.add_argument('--anormly_ratio', type=float, default=4.00)
parser.add_argument('--individual', type=bool, default=False)
parser.add_argument('--cutfreq', type=int, default=25)
parser.add_argument('--DSR', type=int, default=4)
parser.add_argument('--plot', type=bool, default=False)
parser.add_argument('--ema_decay', type=float, default=0.5)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--layer', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--dataset', type=str, default="SMD")
parser.add_argument('--strategy', type=str, default="None")
parser.add_argument('--patience', type=int, default=7)
parser.add_argument('--model', type=str, default="HMLP")

config = parser.parse_args()
torch.cuda.set_device(config.cuda_id)
from utils import SMD_data_split,anoshift_data,Anoshift

def load_data(id=1):
    dataset=config.dataset
    if dataset=="SMD":
        train_raw,test_raw,labels=SMD_data_split(id)
        vali_raw=train_raw[:int(len(train_raw)*0.8)]
        smd_train,smd_test,smd_test_with_label=SMD(train_raw,seq_len=config.win_size),SMD(test_raw,seq_len=config.win_size),SMD(test_raw,labels,seq_len=config.win_size)
        smd_vali=SMD(vali_raw,seq_len=config.win_size)
        train_loader,test_loader,vali_loader=DataLoader(smd_train,batch_size=config.batch_size,shuffle=False),DataLoader(smd_test_with_label),DataLoader(smd_vali,batch_size=config.batch_size)
    elif dataset=="MSL":
        train_raw,test_raw,labels=MSL_SMAP_data(dataset=dataset)
        vali_raw=train_raw[:int(len(train_raw)*0.8)]
        smd_train,smd_test,smd_test_with_label=MSL_SMAP(train_raw,seq_len=config.win_size),MSL_SMAP(test_raw,seq_len=config.win_size),MSL_SMAP(test_raw,labels,seq_len=config.win_size)
        smd_vali=MSL_SMAP(vali_raw,seq_len=config.win_size)
        train_loader,test_loader,vali_loader=DataLoader(smd_train,batch_size=config.batch_size,shuffle=False),DataLoader(smd_test_with_label),DataLoader(smd_vali,batch_size=config.batch_size)
    elif dataset=="SMAP":
        train_raw,test_raw,labels=MSL_SMAP_data(dataset=dataset)
        vali_raw=train_raw[:int(len(train_raw)*0.8)]
        smd_train,smd_test,smd_test_with_label=MSL_SMAP(train_raw,seq_len=config.win_size),MSL_SMAP(test_raw,seq_len=config.win_size),MSL_SMAP(test_raw,labels,seq_len=config.win_size)
        smd_vali=MSL_SMAP(vali_raw,seq_len=config.win_size)
        train_loader,test_loader,vali_loader=DataLoader(smd_train,batch_size=config.batch_size,shuffle=False),DataLoader(smd_test_with_label),DataLoader(smd_vali,batch_size=config.batch_size)
    elif dataset=="PSM":
        train_raw,test_raw,labels=PSM_data()
        vali_raw=train_raw[:int(len(train_raw)*0.8)]
        smd_train,smd_test,smd_test_with_label=PSM(train_raw,seq_len=config.win_size),PSM(test_raw,seq_len=config.win_size),PSM(test_raw,labels,seq_len=config.win_size)
        smd_vali=PSM(vali_raw,seq_len=config.win_size)
        train_loader,test_loader,vali_loader=DataLoader(smd_train,batch_size=config.batch_size,shuffle=False),DataLoader(smd_test_with_label),DataLoader(smd_vali,batch_size=config.batch_size)
    elif dataset=="tesla":
        train_raw,spoof1_raw,label1,spoof2_raw,label2,spoof3_raw,label3,dos_raw,label4,fuzzy_raw,label5=tesla_data()
        vali_raw=train_raw[:int(len(train_raw)*0.8)]
        tesla_train,spoof1,spoof2,spoof3,dos,fuzzy=Tesla(train_raw),Tesla(spoof1_raw,label1),Tesla(spoof2_raw,label2),Tesla(spoof3_raw,label3),Tesla(dos_raw,label4),Tesla(fuzzy_raw,label5)
        tesla_vali=Tesla(vali_raw)
        train_loader,spoof1_loader,spoof2_loader,spoof3_loader,dos_loader,fuzzy_loader=DataLoader(tesla_train,batch_size=config.batch_size,shuffle=False),DataLoader(spoof1,batch_size=config.batch_size,shuffle=False),DataLoader(spoof2,batch_size=config.batch_size,shuffle=False),DataLoader(spoof3,batch_size=config.batch_size,shuffle=False),DataLoader(dos,batch_size=config.batch_size,shuffle=False),DataLoader(fuzzy,batch_size=config.batch_size,shuffle=False)
        vali_loader=DataLoader(tesla_vali,batch_size=config.batch_size,shuffle=False)
        return train_loader,spoof1_loader,spoof2_loader,spoof3_loader,dos_loader,fuzzy_loader,vali_loader
    """
    elif dataset=="Anoshift":
        name=["IID","NEAR","FAR"]
        train_raw,test_raw,labels=anoshift_data(name[id])
        vali_raw=train_raw[:int(len(train_raw)*0.8)]
        smd_train,smd_test,smd_test_with_label=PSM(train_raw,seq_len=config.win_size),PSM(test_raw,seq_len=config.win_size),PSM(test_raw,labels,seq_len=config.win_size)
        smd_vali=PSM(vali_raw,seq_len=config.win_size)
        train_loader,test_loader,vali_loader=DataLoader(smd_train,batch_size=config.batch_size,shuffle=False),DataLoader(smd_test_with_label),DataLoader(smd_vali,batch_size=config.batch_size)
    """
    return train_loader,test_loader,vali_loader,labels

from model import AE,LSTM_AD,HMLP_wo_rin,HMLP_wo_wavelet
model_list={
    "Dlinear":Dlinear,
    "HMLP":HMLP,#(config).cuda(),
    "TFTS":TFTS,#(config).cuda(),
    "AE":AE,
    "LSTM":LSTM_AD,
    "HMLP_wo_rin":HMLP_wo_rin,
    "HMLP_wo_wa":HMLP_wo_wavelet#"AT":AnomalyTransformer(config).cuda(),
}
if config.dataset=="tesla":
    train_loader,spoof1_loader,spoof2_loader,spoof3_loader,dos_loader,fuzzy_loader,vali_loader=load_data(1)
elif config.dataset=="SMD":
    dataset_num=28
    results=[]
    file_mode="a"
    duration=0
    # test 
    file=f"/data/home/Linuo/iot/ILAD/log/wt/{config.dataset}.txt"
    for idx in range(dataset_num):
        train_loader,test_loader,vali_loader,labels=load_data(idx+1)
        #print(idx)
        model=model_list[config.model](config).cuda()
        ema=EMA(model,config.ema_decay)
        ema.register()
        model,_,_=train(model,train_loader,vali_loader,config,"AdamW","MSE")
        #st=time.time()
        #scores,_,_=test(model,test_loader,config,torch.nn.MSELoss())
        #duration+=time.time()-st
        #scores=1
    for idx in range(dataset_num):
        train_loader,test_loader,vali_loader,labels=load_data(idx+1)
        scores,_,_,_=test(model,test_loader,config,torch.nn.MSELoss())
        results.append(scores)
    scores=np.mean(np.array(results),axis=0)
    with open(file,file_mode) as files:
        print(f"continual learning without strategy datasets:{config.dataset},win_size:{config.win_size},model:{config.model},with EMA:{config.ema_decay},with patience:{config.patience},delta:{config.delta},dropout:{config.dropout},layer:{config.layer},epochs:{config.epochs},learning_rate:{config.lr},batch_size:{config.batch_size},weight_decay:{config.wd},duration:{duration},score:{scores}",file=files)
        each_scores=""
        print("len(results)",len(results))
        for i in range(28):
            each_scores+=f"s{i+1}:{results[i]},"
        print(f"winsize:{config.win_size}"+each_scores,file=files)
else:
    model_name=config.model
    model=model_list[model_name](config).cuda()
    train_loader,test_loader,vali_loader,labels=load_data(1)
    ema=EMA(model,config.ema_decay)
    ema.register()
    model,_,_=train(model,train_loader,vali_loader,config,"AdamW","MSE")
torch.save(model.state_dict(),os.path.join("/data/home/Linuo/iot/ILAD/wt",os.path.join(str(config.model),str(config.dataset))+'_checkpoint.pth'))

"""
elif config.dataset=="Anoshift":
    results=[]
    name=["IID","NEAR","FAR"]
    file_mode="a"
    log=f"/data/home/Linuo/iot/ILAD/log/wt/{config.dataset}.txt"
    model_name=config.model
    train_loader,vali_loader,_,_=load_data(0)
    del _
    model=model_list[config.model](config).cuda()
    ema=EMA(model,config.ema_decay)
    ema.register()
    model,_,_=train(model,train_loader,)
    for idx in range(3):
        _,test_loader,labels=load_data(idx)
"""

file_mode="a"
file=f"/data/home/Linuo/iot/ILAD/log/wt/{config.dataset}.txt"

#st=time.time()
if config.dataset=="tesla":
    scores1,_,_=test(model,spoof1_loader,config,torch.nn.MSELoss())
    scores2,_,_=test(model,spoof2_loader,config,torch.nn.MSELoss())
    scores3,_,_=test(model,spoof3_loader,config,torch.nn.MSELoss())
    scores4,_,_=test(model,dos_loader,config,torch.nn.MSELoss())
    scores5,_,_=test(model,fuzzy_loader,config,torch.nn.MSELoss())
    scores=np.mean(np.array([scores1,scores2,scores3,scores4,scores5]),axis=0)

elif config.dataset!="SMD":
    scores,pred,labels_test,duration=test(model,test_loader,config,torch.nn.MSELoss())
#duration=time.time()-st
with open(file,file_mode) as files:
    print(f"continual learning without strategy, datasets:{config.dataset},duration:{duration},win_size:{config.win_size},model:{config.model},with EMA:{config.ema_decay},with patience:{config.patience},delta:{config.delta},dropout:{config.dropout},layer:{config.layer},epochs:{config.epochs},learning_rate:{config.lr},batch_size:{config.batch_size},weight_decay:{config.wd},duration:{duration},score:{scores}",file=files)
    if config.dataset=="tesla":
        print(f"winsize:{config.win_size},spoof1:{scores1},spoof2:{scores2},spoof3:{scores3},dos:{scores4},fuzzy:{scores5}",file=files)
    #elif config.dataset=="SMD":
    #    print(f"winsize:{config.win_size},s1:{scores1},s2:{scores2},s3:{scores3}",file=files)
subject = f"{config.dataset} experiment data"
body = f"datasets:{config.dataset},duration:{duration},win_size:{config.win_size},model:{config.model},with EMA:{config.ema_decay},with patience:{config.patience},delta:{config.delta},dropout:{config.dropout},layer:{config.layer},epochs:{config.epochs},learning_rate:{config.lr},batch_size:{config.batch_size},weight_decay:{config.wd},duration:{duration},score:{scores}"
if config.dataset=="tesla":
    body+=f"winsize:{config.win_size},spoof1:{scores1},spoof2:{scores2},spoof3:{scores3},dos:{scores4},fuzzy:{scores5}"
elif config.dataset=="SMD":
    body+=f"winsize:{config.win_size},score:{scores}"

# 创建一个MIMEText邮件对象，HTML邮件正文
msg = MIMEText(body, 'plain', 'utf-8')
msg['From'] = Header(email_user)
msg['To'] = Header(', '.join(receivers))
msg['Subject'] = Header(subject, 'utf-8')
server = smtplib.SMTP()
try:
    # 连接到SMTP服务器
    server.connect(smtp_server, smtp_port)
    # 登录邮箱
    server.login(email_user, email_pass)
    # 发送邮件
    server.sendmail(sender, receivers, msg.as_string())
    print("邮件发送成功")
except smtplib.SMTPException as e:
    print("Error: 无法发送邮件", e)
finally:
    # 断开服务器连接
    server.quit()
