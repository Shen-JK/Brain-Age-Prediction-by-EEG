import os,sys
import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Function
import numpy as np


class channel_attention(nn.Module):
    def __init__(self, channel,ratio=2):
        super(channel_attention, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(channel, int(channel/ratio), bias=True)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(int(channel/ratio), channel, bias=True)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        self.globalmaxpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc3 = nn.Linear(channel, int(channel/ratio), bias=True)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc3.bias)
        self.fc4 = nn.Linear(int(channel/ratio), channel, bias=True)
        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        avg_pool = self.globalavgpool(x)
        avg_pool = avg_pool.view(x.shape[0], x.shape[1])
        avg_pool = F.relu(self.fc1(avg_pool))
        avg_pool = self.fc2(avg_pool)

        max_pool = self.globalmaxpool(x)
        max_pool = max_pool.view(x.shape[0], x.shape[1])
        max_pool = F.relu(self.fc3(max_pool))
        max_pool = self.fc4(max_pool)

        cbam_feature = torch.add(avg_pool, max_pool)
        cbam_feature = torch.sigmoid(cbam_feature)
        # print(cbam_feature)
        cbam_feature = cbam_feature.view(x.shape[0], x.shape[1], 1, 1)

        out = torch.mul(x, cbam_feature)
        return out
        
class Block_2channel(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block_2channel, self).__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, affine=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels, affine=False)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.avgpool = nn.AvgPool2d(3, (1,2), 1)
                                                                                                                                                                                                                   
    def forward(self, x):
        out1 = F.relu(self.conv1(self.bn1(x)))                                                                                                                                                                     
        out1 = F.relu(self.conv2(self.bn2(out1)))                                                                                                                                                                  
        if self.stride!=1:
            x = self.avgpool(x)                                                                                                                                                                                    
            zeropad = torch.zeros_like(x)                                                                                                                                                                          
            x = torch.cat((x, zeropad), 1)
        out1 =out1+ x                                                                                                                                                                                                  
                                                                                                                                                                                                                   
        out2 = F.relu(self.conv3(self.bn3(out1)))                                                                                                                                                                  
        out2 = F.relu(self.conv4(self.bn4(out2)))                                                                                                                                                                  
        out2 = out2+out1                                                                                                                                                                                               
        return out2


class ResNet_MIT(nn.Module):
    def __init__(self):
        super(ResNet_MIT, self).__init__()
        self.EC=ResNet_2channel_att_forMIT()
        self.EO=ResNet_2channel_att_forMIT()
        self.classifier=nn.Linear(192,21)
        self.dropout=nn.Dropout(0.2)
        self.globalavgpool1=nn.AdaptiveAvgPool2d((1,3))
        self.globalavgpool2=nn.AdaptiveAvgPool2d((1,3))
        self.relu=nn.ReLU()
        self.fusion=iAFF(32,4)
    def forward(self,x):

        x1=x[:,:,:,:153]
        x1=x1.permute(0,1,3,2)
        x2=x[:,:,:,153:]
        x2=x2.permute(0,1,3,2)
        f1,_=self.EC(x1)
        f2,_=self.EO(x2)
        f1=self.globalavgpool1(f1)
        f2=self.globalavgpool1(f2)
        out = torch.flatten(torch.cat((f1,f2),dim=1), start_dim=1)
        out = self.classifier(self.dropout(out))
        out=F.log_softmax(out,dim=1)
        return out

class ResNet_2channel_att_forMIT(nn.Module):
    def __init__(self):
        super(ResNet_2channel_att_forMIT, self).__init__()
        self.res1_1 = self.resnet_layer(1, 28, 3, (1,2), 1, True, False)
        self.res1_2 = self.resnet_layer(1, 28, 3, (1,2), 1, True, False)
        self.block1_1 = Block_2channel(28, 28, 1)
        self.block2_1 = Block_2channel(28, 56, (1,2))
        self.block3_1 = Block_2channel(56, 112, (1,2))
        self.block4_1 = Block_2channel(112, 224, (1,2))
        self.block1_2 = Block_2channel(28, 28, 1)
        self.block2_2 = Block_2channel(28, 56, (1,2))
        self.block3_2 = Block_2channel(56, 112, (1,2))
        self.block4_2 = Block_2channel(112, 224, (1,2))
        self.res2 = self.resnet_layer(224, 448, 3, 1, 1, False, True)
        self.res3 = self.resnet_layer(448, 32, 1, 1, 0, False, True)
        self.bn = nn.BatchNorm2d(32, affine=False)
        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Linear(32,2)
        self.channel_att=channel_attention(32,2)
        self.dropout=nn.Dropout(0.3)
        
    def resnet_layer(self, in_channels, out_channels, kernel_size, stride, padding, learn_bn, use_relu):
        if use_relu:
            return nn.Sequential(nn.BatchNorm2d(in_channels, affine=learn_bn),
                                 nn.ReLU(),                                                                                                                                                                        
                                 nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        else:
            return nn.Sequential(nn.BatchNorm2d(in_channels, affine=learn_bn),
                                 nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    def split_freq1(self, x):
        return x[:,:,:,:64]
    def split_freq2(self, x):
        return x[:,:,:,64:]

    def forward(self, x):

        r1 = self.split_freq1(x)                                                                                                                                                                                   
        r2 = self.split_freq2(x)                                                                                                                                                                                   
        r1 = self.res1_1(r1)
        r1 = self.block1_1(r1)
        r1 = self.block2_1(r1)                                                                                                                                                                                     
        r1 = self.block3_1(r1)                                                                                                                                                                                     
        r1 = self.block4_1(r1)   
        r2 = self.res1_2(r2)
        r2 = self.block1_2(r2)
        r2 = self.block2_2(r2)                                                                                                                                                                                     
        r2 = self.block3_2(r2)                                                                                                                                                                                     
        r2 = self.block4_2(r2)                                                                                                                                                                                     
        out = torch.cat((r1,r2), 3)
        out = self.res2(out)                                                                                                                                                                                       
        
        out = self.res3(out)                                                                                                                                                                                       

        out = self.bn(out)  
        out=self.channel_att(out)
        features=out
        out = self.globalavgpool(out)                                                                                                                                                                              
        out = torch.flatten(out, start_dim=1)
        out=self.dropout(out)
        out=self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return features,out
        
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        # print(xa.shape)
        xg = self.global_att(xa)
        # print(xg.shape)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo