import torch
import torch.nn as nn
from deepod.metrics import ts_metrics
# Importing the necessary libraries and modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import pytorch_wavelets as wavelets
class LSTM_AD(nn.Module):
    def __init__(self,config):
        super(LSTM_AD,self).__init__()
        self.encoder=nn.LSTM(input_size=config.win_size,hidden_size=32,device="cuda",num_layers=4)
        self.decoder=nn.LSTM(input_size=32,hidden_size=config.win_size,device="cuda",num_layers=4)
        #self.bn=nn.BatchNorm1d(num_features=100)
    def forward(self,x):
        #x,h=self.encoder(x)
        return self.decoder(self.encoder(x)[0])[0].permute(0,2,1)


class AE(nn.Module):
    def __init__(self,config):
        super(AE,self).__init__()
        seq_len=config.win_size
        self.encoder=nn.Sequential(nn.Linear(seq_len,seq_len//2),
                                   nn.ReLU(),
                                   nn.Linear(seq_len//2,seq_len//2),
                                   nn.ReLU(),
                                   nn.Linear(seq_len//2,seq_len//2),
                                   nn.ReLU()
                                   )
        self.decoder=nn.Sequential(nn.Linear(seq_len//2,seq_len//2),
                                   nn.ReLU(),
                                   nn.Linear(seq_len//2,seq_len//2),
                                   nn.ReLU(),
                                   nn.Linear(seq_len//2,seq_len),
                                   )
    def forward(self,x):
        return self.decoder(self.encoder(x)).permute(0,2,1)
# Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        assert d_model%num_heads==0,"d_model must be divisible by num_heads"
        #initialize dimensions
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        #embedding of input
        self.W_q=nn.Linear(d_model,d_model)# Query transformation
        self.W_k=nn.Linear(d_model,d_model)# Key transformation
        self.W_v=nn.Linear(d_model,d_model)# Value transformation
        self.W_o=nn.Linear(d_model,d_model)# Output transformation
    
    def scaled_dot_product_attention(self,Q,K,V,mask=None):
        attn_scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None:
            attn_scores=attn_scores.masked_fill(mask==0,-1e9)
        attn_probs=torch.softmax(attn_scores,dim=-1)
        output=torch.matmul(attn_probs,V)
        return output
    
    def split_heads(self,x):
        batch_size,seq_length,d_model=x.size()
        return x.view(batch_size,seq_length,self.num_heads,self.d_k).transpose(1,2)
    def combine_heads(self,x):
        batch_size,_,seq_length,d_k=x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)
    def forward(self,Q,K,V,mask=None):
        Q=self.split_heads(self.W_q(Q))
        K=self.split_heads(self.W_k(K))
        V=self.split_heads(self.W_v(V))
        attn_output=self.scaled_dot_product_attention(Q,K,V,mask)
        output=self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.relu=nn.ReLU()
    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_length):
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(max_seq_length,d_model)
        position=torch.arange(0,max_seq_length,dtype=torch.float).unsqueeze(1)
        #print(position.shape)
        #print(position)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*-(math.log(10000.0)/d_model))
        #print(div_term.shape)
        #print(div_term)
        #print((position*div_term).shape)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        #print(pe.shape)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.size(1)]
class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads)
        self.feed_forward=PositionWiseFeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,mask):
        attn_output=self.self_attn(x,x,x,mask)
        x=self.norm1(x+self.dropout(attn_output))
        ff_output=self.feed_forward(x)
        x=self.norm2(ff_output+self.dropout(ff_output))
        return x
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads)
        self.cross_attn=MultiHeadAttention(d_model,num_heads)
        self.feed_forward=PositionWiseFeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,enc_output,src_mask,tgt_mask):
        attn_output=self.self_attn(x,x,x,tgt_mask)
        x=self.norm1(x+self.dropout(attn_output))
        attn_output=self.cross_attn(x,enc_output,enc_output,src_mask)
        x=self.norm2(x+self.dropout(attn_output))
        ff_output=self.feed_forward(x)
        x=self.norm3(x+self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self,src_vocal_size,tgt_vocal_size,d_model,num_heads,num_layers,d_ff,max_seq_length,dropout):
        super(Transformer,self).__init__()
        self.encoder_embedding=nn.Embedding(src_vocal_size,d_model)
        self.decoder_embedding=nn.Embedding(tgt_vocal_size,d_model)
        self.positional_encoding=PositionalEncoding(d_model,max_seq_length)

        self.encoder_layer=nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.decoder_layer=nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])

        self.fc=nn.Linear(d_model,tgt_vocal_size)
        self.dropout=nn.Dropout(dropout)
    def generate_mask(self,src,tgt):
        src_mask=(src!=0).unsqueeze(1).unsqueeze(2)
        tgt_mask=(tgt!=0).unsqueeze(1).unsqueeze(3)
        seq_length=tgt.size(1)
        nopeak_mask=(1-torch.triu(torch.ones(1,seq_length,seq_length),diagonal=1)).bool()
        tgt_mask=tgt_mask&nopeak_mask
        return src_mask,tgt_mask
    def forward(self,src,tgt):
        src_mask,tgt_mask=self.generate_mask(src,tgt)
        src_embedded=self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded=self.dropout(self.positional_encoding(self.encoder_embedding(tgt)))
        enc_output=src_embedded
        for enc_layer in self.encoder_layer:
            enc_output=enc_layer(enc_output,src_mask)
        dec_output=tgt_embedded
        for dec_layer in self.decoder_layer:
            dec_output=dec_layer(dec_output,enc_output,src_mask,tgt_mask)
        output=self.fc(dec_output)
        return output


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
        #RIN
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
class HMLP_wo_wavelet(nn.Module):
    def __init__(self,config):
        super(HMLP_wo_wavelet,self).__init__()
        self.win_size=config.win_size
        self.layer=config.layer
        self.batch_size=config.batch_size
        self.net=nn.Linear(config.win_size,config.win_size)
        #self.LinearBlock=nn.Sequential(nn.Linear(self.win_size//2,self.win_size//2),
        #                       nn.BatchNorm2d(32),
        #                       nn.ReLU(),
        #                       nn.Dropout(0.2))
    
    def forward(self,x):
        # RIN
        #print("x",x.shape)

        x_mean=torch.mean(x,dim=1,keepdim=True)
        x=x-x_mean
        x_var=torch.var(x,dim=1,keepdim=True)+1e-5
        x=x/torch.sqrt(x_var)
        out=self.net(x)
        out=out*torch.sqrt(x_var)+x_mean
        #print(out.shape)
        #rec_out=out
        #print("rec out",rec_out.shape)
        return out.permute(0,2,1)
## Dlinear
class HMLP_wo_rin(nn.Module):
    def __init__(self,config):
        super(HMLP_wo_rin,self).__init__()
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

        out=out.squeeze(0).permute(0,2,1)#*torch.sqrt(x_var)+x_mean
        #print(out.shape)
        #rec_out=out
        #print("rec out",rec_out.shape)
        return out.permute(0,2,1)

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
class Dlinear(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Dlinear, self).__init__()
        self.seq_len =configs.win_size
        self.pred_len =configs.win_size

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels =configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Decoder = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
       # print(x.shape)
        return x.permute(0,2,1)
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
        return rec_x#,low_xy*torch.sqrt(x_var)

