import numpy as np
import os,math,sys,time
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from random import shuffle
import mne
from torch import Tensor
import random
random.seed(30)

class MyDataset(Dataset):
    def __init__(self, wav_file,batch_size,train_flag=True):
        self.train_flag=train_flag
        self.wavs=open(wav_file,'r').readlines()
        self.batch_size=batch_size
        self.labellis=['4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
        self.offset=[-1,0,1]
    def __len__(self):
        return len(self.wavs)
  
    def __getitem__(self, idx):
        utt=self.wavs[idx].split(' ')[0].split('subj')[1][:4]
        wavname_EC=self.wavs[idx].split(' ')[0]
        wavname_EO=self.wavs[idx].split(' ')[1]
        label=self.wavs[idx].split(' ')[2][:-1]
        label=int(self.labellis.index(label))
        if self.train_flag==True:
            if label==0:
                label=label+random.choice(self.offset[1:])
            elif label==len(self.labellis)-1:
                label=label+random.choice(self.offset[:-1])
            else:
                label=label+random.choice(self.offset)
        elif self.train_flag==False:
            label=label
                
        raw_EC = mne.io.read_raw_fif(wavname_EC)
        t_idx_EC = raw_EC.time_as_index([0., 39.])
        data_EC, times_EC = raw_EC[:, t_idx_EC[0]:t_idx_EC[1]]
        data_EC=np.array(data_EC-np.mean(np.mat(data_EC),1))
        tmp_EC=data_EC
        stddata=np.std(data_EC,1)
        data_EC=data_EC[(stddata>0) & (stddata<0.0009),:]
        if data_EC.shape[0]==0:
            data_EC=np.mean(tmp_EC,0)
        elif data_EC.shape[0]!=0:
            data_EC=np.mean(data_EC,0)
        data_EC=torch.from_numpy(data_EC).float().unsqueeze(0)
        
        raw_EO = mne.io.read_raw_fif(wavname_EO)
        t_idx_EO = raw_EO.time_as_index([0., 19.])
        data_EO, times_EO = raw_EO[:, t_idx_EO[0]:t_idx_EO[1]]
        data_EO=np.array(data_EO-np.mean(np.mat(data_EO),1))
        tmp_EO=data_EO
        stddata=np.std(data_EO,1)
        data_EO=data_EO[(stddata>0) & (stddata<0.0009),:]
        if data_EO.shape[0]==0:
            data_EO=np.mean(tmp_EO,0)
        elif data_EO.shape[0]!=0:
            data_EO=np.mean(data_EO,0)
        data_EO=torch.from_numpy(data_EO).float().unsqueeze(0)
        # print(data_EC.shape)
        # print(data_EC.dtype)
        if data_EC.shape[1]<39*500:
            repeat_time=int(39*500/data_EC.shape[1])
            tmp=data_EC
            for i in range(repeat_time):
                data_EC=torch.cat((data_EC,tmp),1)
            data_EC=data_EC[:,:39*500]
        
        if data_EO.shape[1]<19*500:
            repeat_time=int(19*500/data_EO.shape[1])
            tmp=data_EO
            for i in range(repeat_time):
                data_EO=torch.cat((data_EO,tmp),1)
            data_EO=data_EO[:,:19*500]
        
        mel_spec_EC = torchaudio.transforms.MelSpectrogram(sample_rate=500,f_min=0,f_max=250,n_fft=512,win_length=512,hop_length=128,n_mels=128,normalized=False)(data_EC) 
        mel_spec_EC=torch.log(mel_spec_EC+1e-8)
        
        mel_spec_EO= torchaudio.transforms.MelSpectrogram(sample_rate=500,f_min=0,f_max=250,n_fft=512,win_length=512,hop_length=128,n_mels=128,normalized=False)(data_EO)
        mel_spec_EO=torch.log(mel_spec_EO+1e-8)

        spec=torch.cat((mel_spec_EC,mel_spec_EO),2)
        return utt,spec,label


if __name__== '__main__':
    print('load_data')