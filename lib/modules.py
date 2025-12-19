import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from typing import Optional, List
from torch import Tensor

class _simple_learner(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Conv2d(d_model * 2, d_model, 1, 1)

    def forward(self, f_low, f_high):
        low_size = f_low.shape[2:]
        f2_high = F.interpolate(f_high, size=low_size)

        f2_low = torch.cat([f_low, f2_high], dim=1)
        f2_low = self.mlp(f2_low)
        return f2_low

class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out
        
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM_spatial(nn.Module):
    def __init__(self):
        super(CBAM_spatial, self).__init__()
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.SpatialGate(x)
        return x_out
        

class SE(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
       # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        #max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out #+ max_out
        return self.sigmoid(out)*x
        
class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out
        

class xboundlearnerv2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0):#(128,8,512,0.0)
        super().__init__()

        self.xbl = xboundlearner(d_model,
                                 nhead,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout)
        self.xbl1 = xboundlearner(d_model,
                                  nhead,
                                  dim_feedforward=dim_feedforward,
                                  dropout=dropout)
        self.mlp = nn.Conv2d(d_model * 2, d_model, 1, 1)#(256,128,1,1)

    def forward(self, f_low, f_high, xi_low, xi_high):#features_encoded_2, features_encoded_3, latent_tensor_2, latent_tensor_3
       # print('f_low=',f_low.shape)
       # print('f_high=',f_high.shape)
       # print('xi_low=',xi_low.shape)
       # print('xi_high=',xi_high.shape)        
        f2_low = self.xbl(f_low, xi_high)#features_encoded_2   latent_tensor_3
        f2_high = self.xbl1(f_high, xi_low)#features_encoded_3   latent_tensor_2

        low_size = f2_low.shape[2:]
       # print('low_size=======',low_size)
        f2_high = F.interpolate(f2_high, size=low_size)

        f2_low = torch.cat([f2_low, f2_high], dim=1)
        f2_low = self.mlp(f2_low)
       # print('f2_low + f2_low=', (f2_low + f2_low).shape)  
        return f2_low + f2_low


class xboundlearner(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model,
                                                nhead,
                                                dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU()

    def forward(self, tgt, src):#features_encoded_2=([12, 128, 14, 14])   latent_tensor_3=([12, 1, 128])
        "tgt shape: Batch_size, C, H, W "
        "src shape: Batch_size, 1, C    "

        B, C, h, w = tgt.shape#B=12 C=128 H=14 W=14       K=Q=([12, 32, 56, 56])  tgt=  V([12, 64, 56, 56])
        tgt = tgt.view(B, C, h * w).permute(2, 0, 1)  # shape: L=14x14, B=12 C=128       
       # print('tgt======',tgt.shape)
        src = src.permute(1, 0, 2)  # shape: Q:1, B, C  ([1, 12, 128])
       # print('src======',src.shape)
        fusion_feature = self.cross_attn(query=tgt, key=src, value=src)[0]
        tgt = tgt + self.dropout1(fusion_feature)
        tgt = self.norm1(tgt)

        tgt1 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)
        return tgt.permute(1, 2, 0).view(B, C, h, w)
class xboundlearnerv5(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=512, dropout=0.0):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model,
                                                nhead,
                                                dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU()

    def forward(self, tgt, src):#features_encoded_2=([12, 128, 14, 14])   latent_tensor_3=([12, 1, 128])
        "tgt shape: Batch_size, C, H, W "
        "src shape: Batch_size, 1, C    "
     #   print('y1======',tgt.shape)
     #   print('t2======',src.shape)
        B, C, w, h = tgt.shape#B=12 C=128 H=14 W=14       K=Q=([12, 32, 56, 56])  tgt=  V([12, 64, 56, 56])
        #h=w=math.sqrt(L)
        tgt = tgt.view(B, C, h * w).permute(2, 0, 1)  # shape: L=14x14, B=12 C=128       
     #   print('y2======',tgt.shape)
        src = src.permute(1, 0, 2)  # shape: Q:1, B, C  ([1, 12, 128])
      #  print('t3======',src.shape)
        fusion_feature = self.cross_attn(query=src, key=tgt, value=tgt)[0]
       # fusion_feature = self.cross_attn(query=src, key=tgt, value=tgt)[0]
      #  print('fusion_feature',fusion_feature.shape)
        tgt = tgt + self.dropout1(fusion_feature)     
        tgt = self.norm1(tgt)
      #  print('y3======',tgt.shape)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       # print('y4======',tgt2.shape)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
       # print('y5======',tgt.shape)
      #  print('tgt.permute(1, 2, 0).view(B, C, h, w)=',(tgt.permute(1, 2, 0).view(B, C, h, w)).shape)
        return tgt.permute(1, 2, 0).view(B, C, h, w)

class BoundaryWiseAttentionGate2D(nn.Sequential):
    def __init__(self, in_channels, hidden_channels=None):
        super(BoundaryWiseAttentionGate2D, self).__init__(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False), nn.Conv2d(in_channels, 1, kernel_size=1))

    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        weight = torch.sigmoid(
            super(BoundaryWiseAttentionGate2D, self).forward(x))
        x = x * weight + x
        return x

class BoundaryWiseAttentionGateAtrous2D(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):

        super(BoundaryWiseAttentionGateAtrous2D, self).__init__()

        modules = []

        if hidden_channels == None:
            hidden_channels = in_channels // 2

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_channels,
                          3,
                          padding=1,
                          dilation=1,
                          bias=False), nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_channels,
                          3,
                          padding=2,
                          dilation=2,
                          bias=False), nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_channels,
                          3,
                          padding=4,
                          dilation=4,
                          bias=False), nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)))
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_channels,
                          3,
                          padding=6,
                          dilation=6,
                          bias=False), nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)

        self.conv_out = nn.Conv2d(5 * hidden_channels, 1, 1, bias=False)

    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        weight = torch.sigmoid(self.conv_out(res))
        x = x * weight + x
        return x





