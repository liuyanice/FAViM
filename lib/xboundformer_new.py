from turtle import forward
import torch
import cv2
#import baal
import torchvision
import torch.nn as nn
import pywt
from pytorch_wavelets import DWTForward
import torch.nn.functional as F
from lib.Position_embedding import PositionEmbeddingLearned
from lib.modules import xboundlearner, xboundlearnerv2, _simple_learner, xboundlearnerv5
from lib.vision_transformers import in_scale_transformer
from lib.sampling_points import sampling_points, point_sample
from lib.pvtv2 import pvt_v2_b2  #
import scipy.fft
import seaborn as sns
#from lib.pvtv2 import pvt_v2_b1
#from lib.segformer import mit_b1
#from lib.pvt import pvt_tiny  #
import numpy as np
import matplotlib.pyplot as plt
import math
from .memory import XBM1, XBM2, XBM3, XBM4
from torch.distributions import Normal
from .seed_init import place_seed_points
#from resnest.torch import resnest50
from einops import rearrange, reduce
from typing import List
from timm.models.layers import trunc_normal_
#from mamba_ssm import Mamba
from .CCViM import CCViM
from .vmamba import VSSM
#from efficientnet_pytorch import EfficientNet

def _segm_pvtv2(num_classes, im_num, ex_num, xbound, trainsize): #(1,1,1,1,[224,224])
    backbone = pvt_v2_b2(img_size=trainsize)
    
    #backbone = pvt_tiny(img_size=trainsize)
   # backbone = pvt_v2_b1(img_size=trainsize)
  #  backbone = mit_b1(img_size=trainsize)
   # layer_name = 'layer2'
    
  #  backbone1 = EfficientNet.from_pretrained('efficientnet-b2')
   
    if 1:
        path = r'/lab/ly/YanLiu/BUsegNew/BUsegNew/lib/backbone/pvt_v2_b2.pth'
    #    path = r'/lab/ly/YanLiu/BUsegNew/BUsegNew/lib/backbone/pvt_tiny.pth'
      #  path = r'/home/liuyan/TestMBUSeg/lib/backbone/pvt_v2_b1.pth'
      #  path = r'/lab/ly/YanLiu/BUsegNew/BUsegNew/lib/backbone/mit_b1.pth'
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {
            k: v
            for k, v in save_model.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
    #定义线性分类器
    classifier = _simple_classifier(num_classes)
    #加载pixelDecoderModel
    pixelDecoderModel1 = pixelDecoderModel()
    PointHead1 = PointHead()
    #加载backbone完毕，调用自己的分割网络_SimpleSegmentationModel.
    model = _SimpleSegmentationModel(backbone, classifier, im_num, ex_num,
                                     xbound, pixelDecoderModel1, PointHead1)
    return model
    

def _segm_pvtv3(num_classes, im_num, ex_num, xbound, trainsize): #(1,1,1,1,[224,224])
    #加载backbone完毕，调用自己的分割网络_SimpleSegmentationModel.
    model = _SimpleSegmentationModel1()
    return model

class _simple_classifier(nn.Module):
    #定义四个线性分类器:classifier 通道数192->1,classifier1 通道数128->1,classifier2 通道数128->1,classifier3 通道数128->1
    def __init__(self, num_classes):
        super(_simple_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(192, 64, 1, padding=0, bias=False),  #560
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1))
        self.classifier1 = nn.Sequential(nn.Conv2d(128, num_classes, 1))
        self.classifier2 = nn.Sequential(nn.Conv2d(128, num_classes, 1))
        self.classifier3 = nn.Sequential(nn.Conv2d(128, num_classes, 1))

    def forward(self, feature):
        low_level_feature = feature[0]
        output_feature = feature[1]
        output_feature = F.interpolate(output_feature,
                                       size=low_level_feature.shape[2:],
                                       mode='bilinear',
                                       align_corners=False)
        if self.training:
            return [
                self.classifier(
                    torch.cat([low_level_feature, output_feature], dim=1)),
                self.classifier1(feature[1]),
                self.classifier2(feature[2]),
                self.classifier3(feature[3])
            ]
        else:
            return self.classifier(
                torch.cat([low_level_feature, output_feature], dim=1))
                
class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out
        
# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x                
                
                           
                
class PointHead(nn.Module):
    def __init__(self, in_c=192, num_classes=128, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.k = k
        self.beta = beta
        self.softmax= nn.Softmax(dim=-1)
    def forward(self, x, res2, out):#x=[6,3,224,224]  features[0]=[6, 64, 56, 56]) pixout=([6, 128, 56, 56])
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        points = sampling_points(out, x.shape[-1] // 16, self.k, self.beta)
      #  print('points=====',points.shape)
       
        coarse = point_sample(out, points, align_corners=False)
      #  print('coarse=====',points.shape)
        fine = point_sample(res2, points, align_corners=False)
      #  print('out=====',out.shape)
      #  print('res2=====',res2.shape)
      #  print('fine=====',fine.shape)
      #  print('coarse=====',coarse.shape)
       # feature_representation = torch.cat([coarse, fine], dim=1)
     #   print('feature_representation=====',feature_representation.shape)
       # rend = self.mlp(feature_representation)
        affinity_edge = torch.bmm(coarse, fine.transpose(2, 1)).transpose(2, 1)
       # print('affinity_edge=====',affinity_edge.shape)
        affinity = self.softmax(affinity_edge)
        high_edge_feat = torch.bmm(affinity, coarse)#.transpose(2, 1)
       # print('high_edge_feat=====',high_edge_feat.shape)
        rend_edge = high_edge_feat + fine
      #  print('rend_edge=====',rend_edge.shape)
        feature_representation = torch.cat([rend_edge, coarse], dim=1) 
     #   print('feature_representation=====',feature_representation.shape)
        rend = self.mlp(feature_representation)
        return rend, points


class pixelDecoderModel(nn.Module):
    def __init__(self, features = 64, out_features = 128, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes)+1 ), out_features, kernel_size=1)
        self.relu = nn.ReLU()
 
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)
 
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages]  + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

 
#class pixelDecoderModel(nn.Module):
#    def __init__(self, features=256, out_features=128, sizes=(1, 2, 3, 6)):#self.psp = PSPModule(2048, 1024, (1, 2, 3, 6)) features=2048
#        super(pixelDecoderModel, self).__init__()
#        self.features=[]
#        self.bottleneck = nn.Conv2d(features*4, out_features, kernel_size=1)
#        self.relu = nn.ReLU()

#    def forward(self, x): #feates= f[0]
#        priors = [F.upsample(input = i, size=(56, 56), mode='bilinear') for i in x ]
#        bottle = self.bottleneck(torch.cat(priors, 1))
#        return self.relu(bottle)
        
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=drop_rate),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


        
class SAM(nn.Module):
    def __init__(self, num_in=128, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)#16
        self.num_n = (mids) * (mids)#4*4
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))#(4 4)

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)#  32      16
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)#  32      16
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)# 16     32

    def forward(self, x, edge):#  ([6, 128, 28, 28])        ([6, 128, 56, 56])
     #   edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped =self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
       # print("x_mask",x_mask.shape)
        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))#(1 1 128 ,  H/8 W/8 128)
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.conv_extend((x_state))

        return out

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


class FSEM(nn.Module):
    def __init__(self, in_channels): #, out_channels
        super(FSEM, self).__init__()
       # self.avg_pool = nn.AdaptiveAvgPool2d(1)
      #  self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels=in_channels//2, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False),#groups=1,
                       IBNorm(in_channels//2))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels//2, out_channels=in_channels//2, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False),
                       IBNorm(in_channels//2))

        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels=in_channels//4, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False),
                       IBNorm(in_channels//4))
        
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels//4, out_channels=in_channels//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                       IBNorm(in_channels//4))
        self.conv2_3 = nn.Sequential(nn.Conv2d(in_channels//4, out_channels=in_channels//2, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
                       IBNorm(in_channels//2))
        self.relu = nn.ReLU()
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels=in_channels//4, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                       IBNorm(in_channels//4))
        self.conv3_2 = nn.Sequential(nn.Conv2d(in_channels//4, out_channels=in_channels//4, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
                       IBNorm(in_channels//4))
        self.conv3_3 = nn.Sequential(nn.Conv2d(in_channels//4, out_channels=in_channels//2, kernel_size=3, stride=1, padding=5, dilation=5, bias=False),
                       IBNorm(in_channels//2))
        
        self.conv4 =nn.Sequential( nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                       IBNorm(in_channels))
       # self.bn = nn.BatchNorm2d(in_channels)
        self.bn = IBNorm(in_channels)
        self.psa = nn.Sequential(PVMLayer(in_channels, in_channels),
                  PVMLayer(in_channels, in_channels))
     #   self.psa = SequentialPolarizedSelfAttention(channel=in_channels)
        self.se = SE(in_channels)
        #self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
       # self.relu1 = nn.ReLU()
       # self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

       # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        psa = self.psa(x)
      #  psa = x
   #     print("psa.shape === ",psa.shape)
        x1 = self.conv1_2(self.conv1_1(x))
  #      print("x1.shape === ",x1.shape)
        x1_1 = self.conv1_1(x)
  #      print("x1_1.shape === ",x1_1.shape)
        x1_2 = torch.cat([x1, x1_1], dim=1)
        
        x2 = self.conv2_3(self.conv2_2(self.conv2_1(x)))
     #   print("x2.shape === ",x1.shape)
        x2_1 = self.conv2_1(x)
       # print("x2_1.shape === ",x2_1.shape)
        x2_2 = self.conv2_2(self.conv2_1(x))
  #      print("x2.shape === ",x2.shape)
   #     print("x2_1.shape === ",x2_1.shape)
  #      print("x2_2.shape === ",x2_2.shape)
       
        x2_3 = torch.cat([x2, x2_2, x2_2], dim=1)
        
        x3 = self.conv3_3(self.conv3_2(self.conv3_1(x)))
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(self.conv3_1(x))
        x3_3 = torch.cat([x3, x3_1, x3_2], dim=1)
        
  #      print("x1_2.shape === ",x1_2.shape)
  #      print("x2_3.shape === ",x2_3.shape)
  #      print("x3_3.shape === ",x3_3.shape)
        out = x1_2 + x2_3 + x3_3
  #      print("out.shape === ",out.shape)
        f = self.se(self.bn(out))
        f = self.relu(self.conv4(f+ psa))
  #      print("f======",f.shape)
        #self.sigmoid(out)
     #   max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
     #   out = avg_out + max_out
        return  f
  
class CADM4(nn.Module):
    def __init__(self, dim, head_num=8):
        super(CADM4, self).__init__()
        self.hidden_size = dim 
        self.head_num = head_num
        self.conv = nn.Conv2d(dim, out_channels=128, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
        self.conv1 = nn.Conv2d(dim+192, out_channels=128, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)
      #  self.attention_norm = nn.LayerNorm(dim)
      #  self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_f = nn.Linear(576, 576)  # qkv_h 484  121(352) 576   144(384)
        self.qkv_local_c = nn.Linear(576, 576)  # qkv_v 121  121      144  144
        self.qkv_local_fc1 = nn.Conv2d(128, 128//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = nn.Conv2d(128//4, 128, 1, 1)# nn.Linear(320*4, 320)  # qkv_v
    #    self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
       # self.ffn_norm = nn.LayerNorm(dim)
      #  self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
      #  self.PEG = PEG(dim)
        self.ca = ChannelAttention(128)
        self.depth_wise = nn.Conv2d(128//4, 128//4, kernel_size=3, padding=1, groups=128//4)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        
        
    def forward(self, f, c):#4, 320, 24, 24  4 128 , 24, 24
  #      print("f.shape===",f.shape)
  #      print("c.shape===",c.shape)
        c = self.conv(c)
        f = self.conv1(f)
        c_c = self.ca(c)
  #      print("c_c.shape===",c_c.shape)
        h_f = f
        B_f, C_f, H_f, W_f = h_f.size()
        h_f = h_f.view(B_f, C_f, H_f*W_f)#.permute(0, 2, 1).contiguous()
   #     print("h_f.shape===",h_f.shape)
        Q = self.qkv_local_f(h_f)#.permute(0, 2, 1).contiguous()
       # h_f = h_f.view(B_f, C_f, H_f, W_f)
      #  h_f = h_f.permute(0, 2, 3, 1).contiguous()
      #  h_f = h_f.view(B_f * H_f, W_f, C_f)
       # Q= self.qkv_local_h(h_f)
        
        
        h_c = c
        B_c, C_c, H_c, W_c = h_c.size()
        h_c = h_c.view(B_f, C_c, H_c*W_c)#.permute(0, 2, 1).contiguous()
   #     print("h_c.shape===",h_c.shape)
        K= self.qkv_local_c(h_c)#.permute(0, 2, 1).contiguous()
        
        V =K
  #      print("Q.shape===",Q.shape)
  #      print("K.shape===",K.shape)
  #      print("V.shape===",V.shape)
        Q =Q.permute(0, 2, 1).contiguous()
        K =K.permute(0, 2, 1).contiguous()
        V =V.permute(0, 2, 1).contiguous()
        attn_out = self.attn(Q,K,V)
    #    K= K.permute(0, 2, 1).contiguous()
    #    Q= Q.permute(0, 2, 1).contiguous()
    #    attn_out= attn_out.permute(0, 2, 1).contiguous()
   #     print("attn_out.shape===",attn_out.shape)
 #       print("Q1.shape===",Q.shape)
  #      print("K1.shape===",K.shape)
  #      print("V1.shape===",V.shape)
        
        X_1 = attn_out+K+Q
        X_1= X_1.permute(0, 2, 1).contiguous()
  #      print("X_1===",X_1.shape)
        
        
        X_1 = X_1.view(B_c, C_c, H_c, W_c)
  #      print("X_1===",X_1.shape)
        X_2 = X_1
     #   X_1 = X_1.view(B_c, C_c, H_c*W_c).permute(0, 2, 1).contiguous()
  #      print("X_2===",X_1.shape)
        X_1 = self.relu(self.qkv_local_fc1(X_1))
  #      print("X_2===",X_1.shape)
        X_1 = self.depth_wise(X_1)
        X_1 = self.qkv_local_fc2(X_1)
  #      print("X_1===",X_1.shape)
  #      print("X_2===",X_2.shape)
        X_1_saf = X_1 + X_2
  #      print("c_c.shape===",c_c.shape)
   #     print("X_1_saf.shape===",X_1_saf.shape)
   #     print("f.shape===",f.shape)
      #  X_1 = X_1.permute(0, 2, 1).contiguous()
      #  X_O = X_1.view(B_c, C_c, H_c, W_c)
        x = X_1_saf *c_c*f
  #      print("x.shape===",x.shape)
        return x 
        
class CADM3(nn.Module):
    def __init__(self, dim, head_num=8):
        super(CADM3, self).__init__()
        self.hidden_size = dim 
        self.head_num = head_num
        self.conv = nn.Conv2d(dim, out_channels=128, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
      #  self.attention_norm = nn.LayerNorm(dim)
      #  self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_f = nn.Linear(576, 576)  # qkv_h 484  121(352) 576   144(384)
        self.qkv_local_c = nn.Linear(576, 576)  # qkv_v 121  121      144  144
        self.qkv_local_fc1 = nn.Conv2d(128, 128//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = nn.Conv2d(128//4, 128, 1, 1)# nn.Linear(320*4, 320)  # qkv_v
    #    self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
       # self.ffn_norm = nn.LayerNorm(dim)
      #  self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
      #  self.PEG = PEG(dim)
        self.ca = ChannelAttention(128)
        self.depth_wise = nn.Conv2d(128//4, 128//4, kernel_size=3, padding=1, groups=128//4)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        
        
    def forward(self, f, c):#4, 128, 24, 24   4, 320, 24, 24
  #      print("f.shape===",f.shape)
  #      print("c.shape===",c.shape)
        c = self.conv(c)
        c_c = self.ca(c)#4, 128, 24, 24
  #      print("c_c.shape===",c_c.shape)
        h_f = f
        B_f, C_f, H_f, W_f = h_f.size()
        h_f = h_f.view(B_f, C_f, H_f*W_f)#.permute(0, 2, 1).contiguous()
   #     print("h_f.shape===",h_f.shape)
        Q = self.qkv_local_f(h_f)#.permute(0, 2, 1).contiguous()
       # h_f = h_f.view(B_f, C_f, H_f, W_f)
      #  h_f = h_f.permute(0, 2, 3, 1).contiguous()
      #  h_f = h_f.view(B_f * H_f, W_f, C_f)
       # Q= self.qkv_local_h(h_f)
        
        
        h_c = c
        B_c, C_c, H_c, W_c = h_c.size()
        h_c = h_c.view(B_f, C_c, H_c*W_c)#.permute(0, 2, 1).contiguous()
   #     print("h_c.shape===",h_c.shape)
        K= self.qkv_local_c(h_c)#.permute(0, 2, 1).contiguous()
        
        V =K
  #      print("Q.shape===",Q.shape)
  #      print("K.shape===",K.shape)
  #      print("V.shape===",V.shape)
        Q =Q.permute(0, 2, 1).contiguous()
        K =K.permute(0, 2, 1).contiguous()
        V =V.permute(0, 2, 1).contiguous()
        attn_out = self.attn(Q,K,V)
    #    K= K.permute(0, 2, 1).contiguous()
    #    Q= Q.permute(0, 2, 1).contiguous()
    #    attn_out= attn_out.permute(0, 2, 1).contiguous()
   #     print("attn_out.shape===",attn_out.shape)
 #       print("Q1.shape===",Q.shape)
  #      print("K1.shape===",K.shape)
  #      print("V1.shape===",V.shape)
        
        X_1 = attn_out+K+Q
        X_1= X_1.permute(0, 2, 1).contiguous()
  #      print("X_1===",X_1.shape)
        
        
        X_1 = X_1.view(B_c, C_c, H_c, W_c)
  #      print("X_1===",X_1.shape)
        X_2 = X_1
     #   X_1 = X_1.view(B_c, C_c, H_c*W_c).permute(0, 2, 1).contiguous()
  #      print("X_2===",X_1.shape)
        X_1 = self.relu(self.qkv_local_fc1(X_1))
  #      print("X_2===",X_1.shape)
        X_1 = self.depth_wise(X_1)
        X_1 = self.qkv_local_fc2(X_1)
  #      print("X_1===",X_1.shape)
  #      print("X_2===",X_2.shape)
        X_1_saf = X_1 + X_2
  #      print("c_c.shape===",c_c.shape)
   #     print("X_1_saf.shape===",X_1_saf.shape)
   #     print("f.shape===",f.shape)
      #  X_1 = X_1.permute(0, 2, 1).contiguous()
      #  X_O = X_1.view(B_c, C_c, H_c, W_c)
        x = X_1_saf *c_c*f
  #      print("x.shape===",x.shape)
        return x 



class CADM2(nn.Module):
    def __init__(self, dim, head_num=8):
        super(CADM2, self).__init__()
        self.hidden_size = dim 
        self.head_num = head_num
        self.conv = nn.Conv2d(dim, out_channels=64, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
        self.conv1 = nn.Conv2d(dim*2, out_channels=64, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
      #  self.attention_norm = nn.LayerNorm(dim)
      #  self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_f = nn.Linear(576, 576)  # qkv_h 1936    484(352) 2304   576(384)
        self.qkv_local_c = nn.Linear(9216, 576)  # qkv_v  484     484      576    576
        self.qkv_local_fc1 = nn.Conv2d(64, 64//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = nn.Conv2d(64//4, 64, 1, 1)# nn.Linear(320*4, 320)  # qkv_v
    #    self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
       # self.ffn_norm = nn.LayerNorm(dim)
      #  self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
      #  self.PEG = PEG(dim)
        self.ca = ChannelAttention(64)
        self.depth_wise = nn.Conv2d(64//4, 64//4, kernel_size=3, padding=1, groups=64//4)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        
        
    def forward(self, f, c): #4, 128, 24, 24   4, 64, 96, 96
    #    print("f.shape===",f.shape)
    #    print("c.shape===",c.shape)
        c = self.conv(c)
        f = self.conv1(f)
        c_c = self.ca(c)
    #    print("c_c.shape===",c_c.shape)
        h_f = f
        B_f, C_f, H_f, W_f = h_f.size()
        h_f = h_f.view(B_f, C_f, H_f*W_f)#.permute(0, 2, 1).contiguous()
  #      print("h_f.shape===",h_f.shape)
        Q = self.qkv_local_f(h_f)#.permute(0, 2, 1).contiguous()
       # h_f = h_f.view(B_f, C_f, H_f, W_f)
      #  h_f = h_f.permute(0, 2, 3, 1).contiguous()
      #  h_f = h_f.view(B_f * H_f, W_f, C_f)
       # Q= self.qkv_local_h(h_f)
        
        
        h_c = c
        B_c, C_c, H_c, W_c = h_c.size()
        h_c = h_c.view(B_f, C_c, H_c*W_c)#.permute(0, 2, 1).contiguous()
    #    print("h_c.shape===",h_c.shape)
        K= self.qkv_local_c(h_c)#.permute(0, 2, 1).contiguous()
        
        V =K
    #    print("Q.shape===",Q.shape)
    #    print("K.shape===",K.shape)
    #    print("V.shape===",V.shape)
        Q =Q.permute(0, 2, 1).contiguous()
        K =K.permute(0, 2, 1).contiguous()
        V =V.permute(0, 2, 1).contiguous()
        attn_out = self.attn(Q,K,V)
    #    K= K.permute(0, 2, 1).contiguous()
    #    Q= Q.permute(0, 2, 1).contiguous()
    #    attn_out= attn_out.permute(0, 2, 1).contiguous()
   #     print("attn_out.shape===",attn_out.shape)
   #     print("Q1.shape===",Q.shape)
   #     print("K1.shape===",K.shape)
   #     print("V1.shape===",V.shape)
        
        X_1 = attn_out+K+Q
        X_1= X_1.permute(0, 2, 1).contiguous()
    #    print("X_1===",X_1.shape)
        
        
        X_1 = X_1.view(B_c, C_c, H_f, W_f)
   #     print("X_1===",X_1.shape)
        X_2 = X_1
     #   X_1 = X_1.view(B_c, C_c, H_c*W_c).permute(0, 2, 1).contiguous()
   #     print("X_2===",X_1.shape)
        X_1 = self.relu(self.qkv_local_fc1(X_1))
   #     print("X_2===",X_1.shape)
        X_1 = self.depth_wise(X_1)
        X_1 = self.qkv_local_fc2(X_1)
   #     print("X_1===",X_1.shape)
   #     print("X_2===",X_2.shape)
        X_1_saf = X_1 + X_2
 #       print("c_c.shape===",c_c.shape)
 #       print("X_1_saf.shape===",X_1_saf.shape)
  #      print("f.shape===",f.shape)
      #  X_1 = X_1.permute(0, 2, 1).contiguous()
      #  X_O = X_1.view(B_c, C_c, H_c, W_c)
        x = self.up(self.up(X_1_saf)) *c_c*self.up(self.up(f))
   #     print("x.shape===",x.shape)
        return x 


class CADM1(nn.Module):
    def __init__(self, dim, head_num=8):
        super(CADM1, self).__init__()
        self.hidden_size = dim 
        self.head_num = head_num
        self.conv = nn.Conv2d(dim, out_channels=64, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
      #  self.attention_norm = nn.LayerNorm(dim)
      #  self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_f = nn.Linear(9216, 576)  # qkv_h
        self.qkv_local_c = nn.Linear(576, 576)  # qkv_v
        self.qkv_local_fc1 = nn.Conv2d(64, 64//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = nn.Conv2d(64//4, 64, 1, 1)# nn.Linear(320*4, 320)  # qkv_v
    #    self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
       # self.ffn_norm = nn.LayerNorm(dim)
      #  self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
      #  self.PEG = PEG(dim)
        self.ca = ChannelAttention(64)
        self.depth_wise = nn.Conv2d(64//4, 64//4, kernel_size=3, padding=1, groups=64//4)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        
        
    def forward(self, f, c): #   4, 64, 96, 96  4, 128, 24, 24
    #    print("f.shape===",f.shape)
    #    print("c.shape===",c.shape)
        c = self.conv(c)#4, 64, 24, 24
        c_c = self.ca(c)#4, 64, 24, 24
    #    print("c_c.shape===",c_c.shape)
        h_f = f# 4, 64, 96, 96
        B_f, C_f, H_f, W_f = h_f.size()
        h_f = h_f.view(B_f, C_f, H_f*W_f)#.permute(0, 2, 1).contiguous()
    #    print("h_f.shape===",h_f.shape)
        Q = self.qkv_local_f(h_f)#.permute(0, 2, 1).contiguous()
       # h_f = h_f.view(B_f, C_f, H_f, W_f)
      #  h_f = h_f.permute(0, 2, 3, 1).contiguous()
      #  h_f = h_f.view(B_f * H_f, W_f, C_f)
       # Q= self.qkv_local_h(h_f)
        
        
        h_c = c
        B_c, C_c, H_c, W_c = h_c.size()
        h_c = h_c.view(B_f, C_c, H_c*W_c)#.permute(0, 2, 1).contiguous()
     #   print("h_c.shape===",h_c.shape)
        K= self.qkv_local_c(h_c)#.permute(0, 2, 1).contiguous()
        
        V =K
    #    print("Q.shape===",Q.shape)
    #    print("K.shape===",K.shape)
    #    print("V.shape===",V.shape)
        Q =Q.permute(0, 2, 1).contiguous()
        K =K.permute(0, 2, 1).contiguous()
        V =V.permute(0, 2, 1).contiguous()
        attn_out = self.attn(Q,K,V)
    #    K= K.permute(0, 2, 1).contiguous()
    #    Q= Q.permute(0, 2, 1).contiguous()
    #    attn_out= attn_out.permute(0, 2, 1).contiguous()
    #    print("attn_out.shape===",attn_out.shape)
   #     print("Q1.shape===",Q.shape)
    #    print("K1.shape===",K.shape)
    #    print("V1.shape===",V.shape)
        
        X_1 = attn_out+K+Q
        X_1= X_1.permute(0, 2, 1).contiguous()#4, 64, 24, 24
   #     print("X_1===",X_1.shape)
        
        
        X_1 = X_1.view(B_c, C_c, H_c, W_c)
   #     print("X_1===",X_1.shape)
        X_2 = X_1
     #   X_1 = X_1.view(B_c, C_c, H_c*W_c).permute(0, 2, 1).contiguous()
   #     print("X_2===",X_1.shape)
        X_1 = self.relu(self.qkv_local_fc1(X_1))
    #    print("X_2===",X_1.shape)
        X_1 = self.depth_wise(X_1)
        X_1 = self.qkv_local_fc2(X_1)
    #    print("X_1===",X_1.shape)
    #    print("X_2===",X_2.shape)
        X_1_saf = X_1 + X_2
   #     print("c_c.shape===",c_c.shape)
   #     print("X_1_saf.shape===",X_1_saf.shape)
  #      print("f.shape===",f.shape)
   #     print("f.shape===",f.shape)
      #  X_1 = X_1.permute(0, 2, 1).contiguous()
      #  X_O = X_1.view(B_c, C_c, H_c, W_c)
      
        x = self.up(self.up(X_1_saf *c_c))*f
    #    print("x.shape===",x.shape)
        return x #4, 64, 96, 96
        
        
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)#*x


class Gate_Gap4(nn.Module):
    def __init__(self):
        super(Gate_Gap4, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, padding=0),  # xuyaoxiugai
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
       # self.max_pool = nn.AdaptiveMaxPool2d(1)
        

    def forward(self, x):
        combined = self.avg_pool(x)#+self.max_pool(x)
        gate_weights = self.gate_network(combined)

        return gate_weights

class Gate_Gap3(nn.Module):
    def __init__(self):
        super(Gate_Gap3, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Conv2d(160, 1, kernel_size=1, padding=0),  # xuyaoxiugai
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
       # self.max_pool = nn.AdaptiveMaxPool2d(1)
        

    def forward(self, x):
        combined = self.avg_pool(x)#+self.max_pool(x)
        gate_weights = self.gate_network(combined)

        return gate_weights
        
class Gate_Gap2(nn.Module):
    def __init__(self):
        super(Gate_Gap2, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),  # xuyaoxiugai
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
       # self.max_pool = nn.AdaptiveMaxPool2d(1)
        

    def forward(self, x):
        combined = self.avg_pool(x)#+self.max_pool(x)
        gate_weights = self.gate_network(combined)

        return gate_weights        

class Gate_Gap1(nn.Module):
    def __init__(self):
        super(Gate_Gap1, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0),  # xuyaoxiugai
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
       # self.max_pool = nn.AdaptiveMaxPool2d(1)
        

    def forward(self, x):
        combined = self.avg_pool(x)#+self.max_pool(x)
        gate_weights = self.gate_network(combined)

        return gate_weights 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = self.conv1(avg_out)
        max_out = self.conv2(max_out)
        out = self.sigmoid(avg_out)+self.sigmoid(max_out)
       # out = torch.cat([avg_out, max_out], dim=1)
       # out = self.conv1(out)
       # out = self.sigmoid(out)
        return out#*x
        
class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)
        
    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)

class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]

        if with_ibn:       
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 
        
class LightEfficientASPP(nn.Module):
    def __init__(self, in_channels, dilation_rates=[1, 2, 5], channel_reduction=4): #[6,12,18]
        super(LightEfficientASPP, self).__init__()

        out_channels=in_channels // channel_reduction
        # Channel reduction
        self.channel_reduction_conv = Conv2dIBNormRelu(in_channels, in_channels // channel_reduction, kernel_size=1)
        c1_out=out_channels
        c2_out=c1_out//channel_reduction
        c2_out=c2_out//channel_reduction
        # Depth-wise atrous convolutions with point-wise convolutions
        self.conv3x3_1 = nn.Sequential(
            Conv2dIBNormRelu(c1_out, c1_out, kernel_size=3, padding=dilation_rates[0], dilation=dilation_rates[0], groups=c1_out),
            Conv2dIBNormRelu(c1_out, c1_out, kernel_size=1)
        )
        self.conv3x3_2 = nn.Sequential(
            Conv2dIBNormRelu(out_channels, c2_out, kernel_size=3, padding=dilation_rates[1], dilation=dilation_rates[1], groups=c2_out),
            Conv2dIBNormRelu(c2_out, c2_out, kernel_size=1)
        )
        self.conv3x3_3 = nn.Sequential(
            Conv2dIBNormRelu(out_channels, c2_out, kernel_size=3, padding=dilation_rates[2], dilation=dilation_rates[2], groups=c2_out),
            Conv2dIBNormRelu(c2_out, c2_out, kernel_size=1)
        )

        # Recover the number of channels
        self.recover_channels = Conv2dIBNormRelu(c1_out+c2_out+c2_out, in_channels, kernel_size=1)
       # self.recover_channels = Conv2dIBNormRelu(c1_out+c2_out+c2_out, 1, kernel_size=1)

    def forward(self, x):
        reduced_features = self.channel_reduction_conv(x)
        conv3x3_1 = self.conv3x3_1(reduced_features)
        conv3x3_2 = self.conv3x3_2(reduced_features)
        conv3x3_3 = self.conv3x3_3(reduced_features)
        combined_features = torch.cat([conv3x3_1, conv3x3_2,conv3x3_3], dim=1)
        output = self.recover_channels(combined_features)
        weight = torch.sigmoid(output)
        x = x * weight + x
        return x

class RU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x

class RU_up(nn.Module):
    def __init__(self, out_ch, in_ch, in_ch_skip=0, bilinear=False, with_skip=True):
        super(RU_up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  nn.Upsample hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if in_ch_skip == 0 and with_skip:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = RU_double_conv(in_ch + in_ch_skip, out_ch)
        self.relu = nn.ReLU(inplace=True)
        group_num = 32
        if out_ch % 32 == 0 and out_ch >= 32:
            if out_ch % 24 == 0:
                group_num = 24
        elif out_ch % 16 == 0 and out_ch >= 16:
            if out_ch % 16 == 0:
                group_num = 16
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch + in_ch_skip, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, out_ch))
        self.with_skip = with_skip

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.with_skip:
            diff_x = x2.size()[-2] - x1.size()[-2]
            diff_y = x2.size()[-1] - x1.size()[-1]

            x1 = F.pad(x1, (diff_y, 0, diff_x, 0))
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1
        
class RU_double_conv_MCD(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.5):
        super(RU_double_conv_MCD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            baal.bayesian.dropout.Dropout(p=dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x
class RU_up_MCD(nn.Module):
    def __init__(self, out_ch, in_ch, in_ch_skip=0, bilinear=False, with_skip=True, dropout_rate=0.5):
        super(RU_up_MCD, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  nn.Upsample hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if in_ch_skip == 0 and with_skip:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = RU_double_conv_MCD(in_ch + in_ch_skip, out_ch, dropout_rate=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        group_num = 32
        if out_ch % 32 == 0 and out_ch >= 32:
            if out_ch % 24 == 0:
                group_num = 24
        elif out_ch % 16 == 0 and out_ch >= 16:
            if out_ch % 16 == 0:
                group_num = 16
        # print(out_ch, group_num)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch + in_ch_skip, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, out_ch))
        self.with_skip = with_skip

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.with_skip:
            diff_x = x2.size()[-2] - x1.size()[-2]
            diff_y = x2.size()[-1] - x1.size()[-1]

            x1 = F.pad(x1, (diff_y, 0, diff_x, 0))
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
        
class FSEM(nn.Module):
    def __init__(self, in_channels): #, out_channels
        super(FSEM, self).__init__()
       # self.avg_pool = nn.AdaptiveAvgPool2d(1)
      #  self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1_1 = nn.Conv2d(in_channels, out_channels=in_channels//2, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
        self.conv1_2 = nn.Conv2d(in_channels//2, out_channels=in_channels//2, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels=in_channels//4, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(in_channels//4, out_channels=in_channels//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv2_3 = nn.Conv2d(in_channels//4, out_channels=in_channels//2, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.relu = nn.ReLU()
        self.conv3_1 = nn.Conv2d(in_channels, out_channels=in_channels//4, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv3_2 = nn.Conv2d(in_channels//4, out_channels=in_channels//4, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.conv3_3 = nn.Conv2d(in_channels//4, out_channels=in_channels//2, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.psa = SequentialPolarizedSelfAttention(channel=in_channels)
        self.se = SE(in_channels)
        #self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
       # self.relu1 = nn.ReLU()
       # self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

       # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        psa = self.psa(x)
   #     print("psa.shape === ",psa.shape)
        x1 = self.conv1_2(self.conv1_1(x))
  #      print("x1.shape === ",x1.shape)
        x1_1 = self.conv1_1(x)
  #      print("x1_1.shape === ",x1_1.shape)
        x1_2 = torch.cat([x1, x1_1], dim=1)
        
        x2 = self.conv2_3(self.conv2_2(self.conv2_1(x)))
     #   print("x2.shape === ",x1.shape)
        x2_1 = self.conv2_1(x)
       # print("x2_1.shape === ",x2_1.shape)
        x2_2 = self.conv2_2(self.conv2_1(x))
  #      print("x2.shape === ",x2.shape)
   #     print("x2_1.shape === ",x2_1.shape)
  #      print("x2_2.shape === ",x2_2.shape)
       
        x2_3 = torch.cat([x2, x2_2, x2_2], dim=1)
        
        x3 = self.conv3_3(self.conv3_2(self.conv3_1(x)))
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(self.conv3_1(x))
        x3_3 = torch.cat([x3, x3_1, x3_2], dim=1)
        
  #      print("x1_2.shape === ",x1_2.shape)
  #      print("x2_3.shape === ",x2_3.shape)
  #      print("x3_3.shape === ",x3_3.shape)
        out = x1_2 + x2_3 + x3_3
  #      print("out.shape === ",out.shape)
        f = self.se(self.bn(out))
        f = self.relu(self.conv4(f+ psa))
  #      print("f======",f.shape)
        #self.sigmoid(out)
     #   max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
     #   out = avg_out + max_out
        return  f 

class Intra_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)
    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
        feature_h = feature_h.view(B * H, W, C//2)
        feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
        feature_v = feature_v.view(B * W, H, C//2)
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]

        if H == W:
            query = torch.cat((q_h, q_v), dim=0)
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(q_h, k_h, v_h)
            attention_output_v = self.attn(q_v, k_v, v_v)
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x
        
class Inter_SA(nn.Module):
    def __init__(self,dim, head_num):
        super(Inter_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_v
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp(self.hidden_size)
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = torch.chunk(self.conv_h(x_input[0]), 3, dim=1)
        feature_v = torch.chunk(self.conv_v(x_input[1]), 3, dim=1)
        query_h, key_h, value_h = feature_h[0], feature_h[1], feature_h[2]
        query_v, key_v, value_v = feature_v[0], feature_v[1], feature_v[2]

        horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0)
        horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()
        horizontal_groups = horizontal_groups.view(3*B, H, -1)
        horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)
        query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]

        vertical_groups = torch.cat((query_v, key_v, value_v), dim=0)
        vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
        vertical_groups = vertical_groups.view(3*B, W, -1)
        vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
        query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]


        if H == W:
            query = torch.cat((query_h, query_v), dim=0)
            key = torch.cat((key_h, key_v), dim=0)
            value = torch.cat((value_h, value_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)
            attention_output_v = self.attn(query_v, key_v, value_v)
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x
 
class DB5DWTLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DB5DWTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create convolutional layers for low and high frequency
      #  self.low_freq_conv = nn.Conv2d(1, out_channels, 1, bias=False).cuda()
     #   self.high_freq_conv = nn.Conv2d(3, out_channels, 1, bias=False).cuda()

        # Create non-subsampled discrete wavelet transform layer
       # self.dwt = DWTForward(J=1, wave='db5', mode='zero').cuda()#zero symmetric
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
       
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_channels*3, out_channels, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
       # self.denoiser = FrequencyDenoisingModule(in_channels, out_channels)
    def forward(self, x):
    # Perform non-subsampled 2D discrete wavelet transform (on GPU)
        #dwt_output = self.dwt(x)    
        yL, yH = self.dwt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
     #   yH = self.denoiser(yH)
        low_freq = self.outconv_bn_relu_L(yL)
        high_freq = self.outconv_bn_relu_H(yH)

        return low_freq, high_freq#, x_reconstructed
        
class DTCWTFilter(nn.Module):
    def __init__(self, in_channels, out_channels, levels=3):
        super(DTCWTFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 创建用于低频和高频卷积的层
        self.low_freq_conv = nn.Conv2d(1*384, out_channels, 1, bias=False).cuda()
        self.high_freq_conv = nn.Conv2d(3*384, out_channels, 1, bias=False).cuda()

    def forward(self, x):
        # 将输入从CUDA张量转换为NumPy数组
        print("x.shape====", x.shape) 
        x_np = x.detach().cpu().numpy()

        # 执行二维离散小波变换 (在CPU上)
        cA, (cH, cV, cD) = pywt.dwt2(x_np, 'db5', 'symmetric')

        # 将结果转换为PyTorch张量 (在GPU上)
        cA = torch.from_numpy(cA).float().cuda()
        cH = torch.from_numpy(cH).float().cuda()
        cV = torch.from_numpy(cV).float().cuda()
        cD = torch.from_numpy(cD).float().cuda()
    #    print("cA.shape====", cA.shape)
    #    print("cH.shape====", cH.shape)
    #    print("cV.shape====", cV.shape)
   #     print("cD.shape====", cD.shape)
        # 分别处理低频和高频部分 (在GPU上)
        low_freq = self.low_freq_conv(cA)
        high_freq = self.high_freq_conv(torch.cat([cH, cV, cD], dim=1))

        # 执行二维离散小波逆变换 (在CPU上)
        x_reconstructed = pywt.idwt2((cA.cpu().numpy(), (cH.cpu().numpy(), cV.cpu().numpy(), cD.cpu().numpy())), 'db5', 'symmetric')
        x_reconstructed = torch.from_numpy(x_reconstructed).float().cuda()

        return low_freq, high_freq, x_reconstructed


class CADM(nn.Module):
    def __init__(self, dim, head_num=8):
        super(CADM, self).__init__()
        self.hidden_size = dim 
        self.head_num = head_num
        self.conv = nn.Conv2d(dim, out_channels=128, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
        self.conv1 = nn.Conv2d(dim, out_channels=128, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)
      #  self.attention_norm = nn.LayerNorm(dim)
      #  self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_f = nn.Linear(9216, 2304)  # qkv_h
        self.qkv_local_c = nn.Linear(9216, 2304)  # qkv_v
        self.qkv_local_fc1 = nn.Conv2d(128, 128//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = nn.Conv2d(128//4, 128, 1, 1)# nn.Linear(320*4, 320)  # qkv_v
    #    self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
       # self.ffn_norm = nn.LayerNorm(dim)
      #  self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
      #  self.PEG = PEG(dim)
        self.ca = ChannelAttention(128)
        self.depth_wise = nn.Conv2d(128//4, 128//4, kernel_size=3, padding=1, groups=128//4)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        
        
    def forward(self, f, c):# 4, 384, 96, 96
    #    print("f.shape===",f.shape)
    #    print("c.shape===",c.shape)
        c = self.conv(c)# 4, 128, 96, 96
        f = self.conv1(f)
        c_c = self.ca(c)#4, 128, 96, 96
    #    print("c_c.shape===",c_c.shape)
        h_f = f#4, 384, 96, 96
        B_f, C_f, H_f, W_f = h_f.size()#4, 128, 96, 96
        h_f = h_f.view(B_f, C_f, H_f*W_f)#4, 128, 9216
  #      print("h_f.shape===",h_f.shape)
        Q = self.qkv_local_f(h_f)#4, 128, 2304
       # h_f = h_f.view(B_f, C_f, H_f, W_f)
      #  h_f = h_f.permute(0, 2, 3, 1).contiguous()
      #  h_f = h_f.view(B_f * H_f, W_f, C_f)
       # Q= self.qkv_local_h(h_f)
        
        
        h_c = c# 4, 128, 96, 96
        B_c, C_c, H_c, W_c = h_c.size()# 4, 128, 96, 96
        h_c = h_c.view(B_f, C_c, H_c*W_c)#   4, 128, 9216                        .permute(0, 2, 1).contiguous()
    #    print("h_c.shape===",h_c.shape)
        K= self.qkv_local_c(h_c)# 4, 128, 2304           .permute(0, 2, 1).contiguous()
        
        V =K#4, 128, 2304
    #    print("Q.shape===",Q.shape) 4, 128, 2304
    #    print("K.shape===",K.shape) 4, 128, 2304
    #    print("V.shape===",V.shape) 4, 128, 2304
        Q =Q.permute(0, 2, 1).contiguous()#
        K =K.permute(0, 2, 1).contiguous()
        V =V.permute(0, 2, 1).contiguous()
        attn_out = self.attn(Q,K,V)
    #    K= K.permute(0, 2, 1).contiguous()
    #    Q= Q.permute(0, 2, 1).contiguous()
    #    attn_out= attn_out.permute(0, 2, 1).contiguous()
   #     print("attn_out.shape===",attn_out.shape)
   #     print("Q1.shape===",Q.shape)
   #     print("K1.shape===",K.shape)
   #     print("V1.shape===",V.shape)
        
        X_1 = attn_out+K+Q
        X_1= X_1.permute(0, 2, 1).contiguous()
    #    print("X_1===",X_1.shape)
        
        
        X_1 = X_1.view(B_c, C_c, H_c//2, W_c//2)# 4, 128, 48, 48
   #     print("X_1===",X_1.shape)
        X_2 = X_1
     #   X_1 = X_1.view(B_c, C_c, H_c*W_c).permute(0, 2, 1).contiguous()
   #     print("X_2===",X_1.shape)
        X_1 = self.relu(self.qkv_local_fc1(X_1))
   #     print("X_2===",X_1.shape)
        X_1 = self.depth_wise(X_1)
        X_1 = self.qkv_local_fc2(X_1)# 4, 128, 48, 48
   #     print("X_1===",X_1.shape)
   #     print("X_2===",X_2.shape)
        X_1_saf = X_1 + X_2   # 4, 128, 48, 48
  #      print("c_c.shape===",c_c.shape)
   #     print("X_1_saf.shape===",X_1_saf.shape)
  #      print("f.shape===",f.shape)
      #  X_1 = X_1.permute(0, 2, 1).contiguous()
      #  X_O = X_1.view(B_c, C_c, H_c, W_c)
        x = (self.up(X_1_saf) *c_c)*f
   #     print("x.shape===",x.shape)
        return x # 4, 384, 96, 96

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y
    
class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq


        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
            
class MFMSAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 groups=32):
        super(MFMSAttentionBlock, self).__init__()

        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx, dilation=1 + scale_idx, groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True)
            ))

        c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(inter_channel, c2wh[in_channels], c2wh[in_channels], frequency_branches, frequency_selection)))
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)))

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            feature = self.multi_scale_branches[scale_idx](feature)
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)
            feature = self.multi_frequency_branches_conv2[scale_idx](feature * (1 - spatial_attention_map) * self.alpha_list[scale_idx] + feature * spatial_attention_map * self.beta_list[scale_idx])
            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2**scale_idx, mode='bilinear', align_corners=None) if (x.shape[2] != feature.shape[2]) or (x.shape[3] != feature.shape[3]) else feature
        feature_aggregation /= self.scale_branches
        feature_aggregation += x

        return feature_aggregation
class LightConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 新增通道扩展层
        self.expand_ratio = out_ch // in_ch
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*self.expand_ratio, 3, padding=1, groups=in_ch),  # groups=in_ch
            nn.GELU(),
            SEBlock(in_ch*self.expand_ratio),
            nn.Conv2d(in_ch*self.expand_ratio, out_ch, 1)  # 点卷积调整通道
        )
        self.dilation_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(4, out_ch)
        )
    
    def forward(self, x):
        x = self.dw_conv(x)
        return self.dilation_conv(x) + x

class SEBlock(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//ratio),
            nn.ReLU(),
            nn.Linear(ch//ratio, ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        
class FlexibleLightConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw_conv = nn.Sequential(
            # 第一阶段：深度卷积
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.GELU(),
            SEBlock(in_ch),
            # 第二阶段：通道扩展
            nn.Conv2d(in_ch, out_ch, 1)
        )
        self.dilation_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(4, out_ch)
        )
    
    def forward(self, x):
        x = self.dw_conv(x)
        return self.dilation_conv(x) + x        
        
def get_gabor_filter(kernel_size, theta, sigma=2.0, lambd=1.0, gamma=0.5, psi=0):
    """生成Gabor滤波器核"""
    radius = kernel_size // 2
    x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    # 旋转坐标
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    # 计算Gabor核
    gb = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * \
         np.cos(2 * np.pi * x_theta / lambd + psi)
    
    # 归一化到[-1, 1]范围
    gb = gb / np.max(np.abs(gb))
    return gb.astype(np.float32)

class BidirectionalGateFusion(nn.Module):
    """双向门控融合模块"""
    def __init__(self, channels):
        super().__init__()
        # 门控生成器
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2*channels, channels//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels//2, 2, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, feat_mamba, feat_entropy):
        # 拼接特征生成门控
        concat_feat = torch.cat([feat_mamba, feat_entropy], dim=1)
        gates = self.gate_conv(concat_feat)  # [B,2,C,H,W]
        gate_mamba, gate_entropy = gates.chunk(2, dim=1)
        
        # 双向门控融合
        fused_mamba = feat_mamba * gate_mamba + feat_entropy * (1 - gate_mamba)
        fused_entropy = feat_entropy * gate_entropy + feat_mamba * (1 - gate_entropy)
        
        return fused_mamba + fused_entropy

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
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

class LightweightDynamicRouter(nn.Module):
    """修正后的动态路由控制器（维度安全）"""
    def __init__(self, in_dim: int, num_branches: int = 3, reduction: int = 4):
        super().__init__()
        self.num_branches = num_branches
        
        # 维度提升确保≥4通道
        self.safe_dim = max(in_dim, 3)
      #  self.origin_proj = nn.Conv2d(in_dim, self.safe_dim, 1)
        
        # 通道门控（输入通道数修正）
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.safe_dim*(num_branches), max(1, self.safe_dim//reduction), 1),
            nn.ReLU(),
            nn.Conv2d(max(1, self.safe_dim//reduction), num_branches, 1),
            nn.Softmax(dim=1)
        )
        
        # 空间注意力（使用安全维度）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(self.safe_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, origin: torch.Tensor, mamba_features: List[torch.Tensor]) -> torch.Tensor:
        # 统一特征维度
        #origin = self.origin_proj(origin)
        mamba_features = [self._adapt_dim(feat, self.safe_dim) for feat in mamba_features]
        
        # 合并特征（通道维度拼接）
        all_features = mamba_features  #+origin
        concatenated = torch.cat(all_features, dim=1)  # [B, safe_dim*(num_branches+1), H, W]
        
        # 通道权重
        c_weights = self.channel_gate(concatenated)  # [B, num_branches+1, 1, 1]
        
        # 空间权重
        s_weights = self.spatial_attn(origin)  # [B, 1, H, W]
        
        # 拆分特征
        split_features = torch.split(concatenated, self.safe_dim, dim=1)  # 拆分成num_branches+1个
        
        # 动态融合
        fused = 0
        alpha = torch.sigmoid(self.alpha)
        for i, feat in enumerate(split_features):
            # 通道权重分量
            c_weight = c_weights[:, i].view(-1, 1, 1, 1)
            
            # 空间权重复用
            s_weight = s_weights if i == 0 else s_weights.detach()
            
            # 特征融合
            fused += feat * (alpha * c_weight + (1 - alpha) * s_weight)
        
        return fused

    def _adapt_dim(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """维度适配器"""
        if x.shape[1] != target_dim:
            return nn.functional.conv2d(x, 
                torch.eye(target_dim, x.shape[1], device=x.device).view(target_dim, x.shape[1], 1, 1)
            )
        return x
        
class DynamicScanner(nn.Module):
    """Lightweight Dynamic Scanner supporting various scan modes."""
    def __init__(self, mode: str, window_size: int = 8):
        super().__init__()
        self.mode = mode
        self.ws = window_size  # Window size for block scanning
        
        # Parameter for spiral direction control
        if mode == 'spiral':
            self.direction = nn.Parameter(torch.tensor([1.0, -1.0]))

    def _get_line_scan(self, h: int, w: int) -> torch.Tensor:
        """Generates line scan coordinates from top-left to bottom-right."""
        return torch.stack([torch.arange(h*w) // w, torch.arange(h*w) % w], dim=1)

    def _get_column_scan(self, h: int, w: int) -> torch.Tensor:
        """Generates column scan coordinates from top-left to bottom-right."""
        return torch.stack([torch.arange(h*w) % h, torch.arange(h*w) // h], dim=1)

    def _generate_spiral(self, h: int, w: int) -> torch.Tensor:
        """Generates spiral scan coordinates starting from center."""
        x, y = h // 2, w // 2
        coords = []
        visited = set()
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        step_size = 1
        current_dir = 0
        
        dir_tensor = torch.sigmoid(self.direction) * 2 - 1
        clockwise = dir_tensor[0] > 0
        
        while len(coords) < h * w:
            for _ in range(2):
                dx, dy = directions[current_dir]
                for _ in range(step_size):
                    if 0 <= x < h and 0 <= y < w and (x, y) not in visited:
                        coords.append((x, y))
                        visited.add((x, y))
                    x += dx
                    y += dy
                    if len(coords) >= h * w:
                        break
                if len(coords) >= h * w:
                    break
                current_dir = (current_dir + (1 if clockwise else -1)) % 4
            step_size += 1
        
        return torch.tensor(coords, dtype=torch.long)

    def _get_block_scan(self, h: int, w: int) -> torch.Tensor:
        """Generates block scan coordinates by dividing the image into windows."""
        num_h, num_w = (h + self.ws - 1) // self.ws, (w + self.ws - 1) // self.ws
        scans = []
        
        for i in range(num_h):
            for j in range(num_w):
                h_start, w_start = i * self.ws, j * self.ws
                h_end, w_end = min(h_start + self.ws, h), min(w_start + self.ws, w)
                rows = torch.arange(h_start, h_end)
                cols = torch.arange(w_start, w_end)
                grid = torch.stack(torch.meshgrid(rows, cols, indexing='ij'), dim=-1)
                scans.append(grid.reshape(-1, 2))
        
        return torch.cat(scans, dim=0).long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        device = x.device
        
        if self.mode == 'line':
            coords = self._get_line_scan(h, w)
        elif self.mode == 'block':
            coords = self._get_block_scan(h, w)
        elif self.mode == 'spiral':
            coords = self._generate_spiral(h, w)
        elif self.mode == 'column':
            coords = self._get_column_scan(h, w)
        else:
            raise ValueError(f"Unsupported scan mode: {self.mode}")
        
        return coords.to(device)


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):#4, 12544, 32
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
       # B, _ = x.shape[:2]
      #  assert C == self.input_dim
     #   n_tokens = x.shape[2:].numel()
    #    img_dims = x.shape[2:]
     #   x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale *x1#self.skip_scale *
        x_mamba2 = self.mamba(x2) + self.skip_scale *x2 
        x_mamba3 = self.mamba(x3) + self.skip_scale *x3 
        x_mamba4 = self.mamba(x4) + self.skip_scale *x4 
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)
        
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
    #    print("x_mamba.shape==", x_mamba.shape)
      #  out = x_mamba#.transpose(-1, -2)#.reshape(B, self.output_dim, *img_dims)
   #     print("out.shape==", out.shape)
        return x_mamba


class LightPVM(nn.Module):
    """轻量级PVM模块（含空间注意力加权融合）"""
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int,
                 scan_modes: List[str] = ['line', 'column', 'block', 'spiral'],
                 img_size: int = 224,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_paths = len(scan_modes)
        
        # 多路径扫描器
        self.scanners = nn.ModuleList([
            DynamicScanner(mode=mode) for mode in scan_modes
        ])
        
        # PVMLayer 并行处理
      #  self.pvm_layers = nn.ModuleList([
      #      PVMLayer(in_dim, out_dim, d_state, d_conv, expand) 
      #      for _ in scan_modes
      #  ])
        self.pvm_layer = PVMLayer(in_dim, out_dim, d_state, d_conv, expand) 
        # 原始特征处理器
      #  self.origin_processor = nn.Sequential(
      #      nn.Conv2d(in_dim, out_dim, 3, padding=1),
      #      nn.GELU()
     #   )
        
        # 空间注意力生成器
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=1)
        
        # 最终投影层
   #     self.proj = nn.Sequential(
   #         nn.Conv2d(out_dim, out_dim, 1),
   #         nn.GELU()
   #     )

    def _apply_scan(self, x: torch.Tensor, scanner: DynamicScanner) -> torch.Tensor:
        B, C, H, W = x.shape
        coords = scanner(x)
        seq = x.permute(0, 2, 3, 1).contiguous()
        scanned = seq[:, coords[:, 0], coords[:, 1]]
        return scanned, seq, coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 步骤1：生成空间注意力图
        #origin_feat = self.origin_processor(x)  # [B, out_dim, H, W]
        origin_feat = x
        # 通道聚合（最大池化+平均池化）
        max_pool = torch.max(origin_feat, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        avg_pool = torch.mean(origin_feat, dim=1, keepdim=True)    # [B, 1, H, W]
        combined = torch.cat([max_pool, avg_pool], dim=1)          # [B, 2, H, W]
        
        # 生成注意力权重
        spatial_weights = self.attention_conv(combined)  # [B, 1, H, W]
        spatial_weights = torch.sigmoid(spatial_weights)  # [0,1]范围
        
        # 步骤2：多路径特征提取
        path_outputs = []
      #  for scanner, pvm_layer in zip(self.scanners, self.pvm_layers):
        for scanner in self.scanners:
            scanned_seq, seq, coords = self._apply_scan(x, scanner)
            pvm_out = self.pvm_layer(scanned_seq)
            
            # 重建特征图
            output = torch.zeros((B, H, W, self.out_dim), 
                                device=x.device,
                                dtype=pvm_out.dtype)
            output[:, coords[:,0], coords[:,1]] = pvm_out
            output = output.permute(0, 3, 1, 2).contiguous()  # [B, out_dim, H, W]
            path_outputs.append(output)
        
        # 步骤3：注意力加权融合
        weighted_outputs = [path * spatial_weights for path in path_outputs]
        fused = sum(weighted_outputs)  # [B, out_dim, H, W]
        return fused
       # return self.proj(fused)
        
class SpatialEntropyFilter(torch.nn.Module):
    def __init__(self, radius=2, bins=10, epsilon=1e-6):
        super().__init__()
        self.radius = radius
        self.bins = bins
        self.epsilon = epsilon
        self.kernel_size = 2 * radius + 1
        
        # 预生成圆形掩膜和有效像素数
        u, v = torch.meshgrid(torch.arange(self.kernel_size), 
                             torch.arange(self.kernel_size), indexing='ij')
        mask = ((u - self.radius)**2 + (v - self.radius)**2 <= self.radius**2).float()
        self.register_buffer('mask', mask.view(1, 1, -1))  # [1,1,k²]
        self.register_buffer('valid_pixels', mask.sum())   # 标量

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size
        
        # 1. 滑动窗口展开
        x_unfold = F.unfold(x, k, padding=self.radius, stride=1)  # [B, C*k², L]
        x_unfold = x_unfold.view(B, C, k*k, H, W)                 # [B,C,k²,H,W]
        
        # 2. 应用掩膜并分箱
        masked = x_unfold * self.mask.unsqueeze(-1).unsqueeze(-1) # [B,C,k²,H,W]
        bin_indices = (masked * (self.bins - 1)).long()           # [B,C,k²,H,W]
        
        # 3. 向量化直方图统计
        hist = torch.zeros((B, C, self.bins, H, W), device=x.device)
        for b in range(self.bins):
            hist[:, :, b] = (bin_indices == b).sum(dim=2)  # [B,C,H,W]
        
        # 4. 概率与熵计算
        prob = hist / (self.valid_pixels + self.epsilon)    # [B,C,bins,H,W]
        entropy = -torch.sum(prob * torch.log(prob + self.epsilon), dim=2)  # [B,C,H,W]
        
        return entropy

class FastEntropy(nn.Module):
    def __init__(self, radius=5, bins=32):
        super().__init__()
        self.radius = radius
        self.bins = bins
        self.window = 2*radius+1
        
        # 预计算基础圆形掩膜 (单通道)
        y, x = torch.meshgrid(torch.arange(self.window), torch.arange(self.window))
        self.base_mask = ((x-radius)**2 + (y-radius)**2 <= radius**2).float()
        self.base_mask = self.base_mask.view(1, 1, self.window, self.window)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 动态生成多通道掩膜
        mask = self.base_mask.repeat(C, 1, 1, 1).to(x.device)  # [C,1,window,window]
        
        # 填充输入
        x_pad = F.pad(x, (self.radius,)*4, mode='reflect')
        
        # 初始化直方图张量
        hist = torch.zeros(B, C, self.bins, H, W, device=x.device)
        
        # 计算每个bin的统计量
        quantized = (x_pad * (self.bins-1)).long()
        for b in range(self.bins):
            bin_val = (quantized == b).float()
            bin_sum = F.conv2d(bin_val, mask, padding=0, groups=C)
            hist[:, :, b] = bin_sum[:, :, :H, :W]
        
        # 计算概率和熵
        prob = hist / mask[0].sum()
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=2)
        return entropy  # 关键修改：返回所有通道的熵 [B,C,H,W]
      
class FACLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FACLayer, self).__init__()
        # 非对称卷积层（处理频域实虚合并通道）
        self.conv_v = nn.Conv2d(2*in_channels, 2*out_channels, (1, kernel_size),
                               padding=(0, kernel_size//2))
        self.conv_h = nn.Conv2d(2*in_channels, 2*out_channels, (kernel_size, 1),
                               padding=(kernel_size//2, 0))

    def forward(self, x):
        # 1. 频域转换
        x_fft = torch.fft.fft2(x)
        x_real, x_imag = x_fft.real, x_fft.imag
        
        # 2. 频谱投影 (VAP/HAP)
        # 合并实部虚部作为双通道
        x_freq = torch.cat([x_real, x_imag], dim=1)  # [B, 2C, H, W]
        
        # 垂直平均池化 (B,2C,H,W) -> (B,2C,1,W)
        vap = F.adaptive_avg_pool2d(x_freq, (1, None))  
        # 水平平均池化 (B,2C,H,W) -> (B,2C,H,1)
        hap = F.adaptive_avg_pool2d(x_freq, (None, 1))

        # 3. 非对称卷积
        vap_conv = self.conv_v(vap)  # 垂直卷积 (1xk)
        hap_conv = self.conv_h(hap)  # 水平卷积 (kx1)

        # 上采样到原尺寸
        vap_upsampled = F.interpolate(vap_conv, size=x_freq.shape[-2:], 
                                    mode='bilinear', align_corners=False)
        hap_upsampled = F.interpolate(hap_conv, size=x_freq.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # 4. 特征聚合
        aggregated = vap_upsampled + hap_upsampled
        
        # 分离实虚并逆变换
        out_real, out_imag = torch.chunk(aggregated, 2, dim=1)
        return torch.fft.ifft2(torch.complex(out_real, out_imag)).real
class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)


class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
 
class LightGaborConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 sigma=2.0, lambd=1.0, gamma=0.5):
        super(LightGaborConv, self).__init__()
        # Depthwise卷积（分组卷积）
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=kernel_size//2, groups=in_channels, bias=False
        )
        # Pointwise卷积（1x1卷积调整通道）
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化Depthwise卷积权重为Gabor滤波器
        self._init_gabor_weights(kernel_size, in_channels, sigma, lambd, gamma)
        
    def _init_gabor_weights(self, kernel_size, in_channels, sigma, lambd, gamma):
        # 为每个通道生成不同方向的Gabor滤波器
        weights = []
        for i in range(in_channels):
            theta = torch.rand(1).item() * math.pi  # 随机方向（0 ~ π）
            gb = get_gabor_filter(kernel_size, theta, sigma, lambd, gamma)
            gb_tensor = torch.from_numpy(gb).view(1, 1, kernel_size, kernel_size)
            weights.append(gb_tensor)
        # 合并权重并设置为可训练参数
        self.depthwise.weight = nn.Parameter(torch.cat(weights, dim=0))
        
    def forward(self, x):
        x = self.depthwise(x)  # Depthwise卷积（Gabor特征提取）
        x = self.pointwise(x)  # 通道调整
        x = self.bn(x)         # 批归一化
        x = self.relu(x)       # 激活函数
        return x

class LightGaborConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(LightGaborConv1, self).__init__()
        # Depthwise convolution (grouped convolution)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=kernel_size//2, groups=in_channels, bias=False
        )
        # Pointwise convolution (1x1 convolution for channel adjustment)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize depthwise convolution weights without Gaussian kernel
   #     self._init_gabor_weights(kernel_size, in_channels)
        
   # def _init_gabor_weights(self, kernel_size, in_channels):
        # Initialize weights for each channel with random values or a simple pattern
   #     weights = []
   #     for i in range(in_channels):
            # Random initialization or use a simple pattern (e.g., identity, Sobel filters)
    #        weight = torch.rand(kernel_size, kernel_size)  # Random initialization
    #        weight_tensor = weight.view(1, 1, kernel_size, kernel_size)
    #        weights.append(weight_tensor)
        
        # Combine weights and set them as trainable parameters
      #  self.depthwise.weight = nn.Parameter(torch.cat(weights, dim=0))
        
    def forward(self, x):
        x = self.depthwise(x)  # Depthwise convolution (Gabor-like feature extraction)
        x = self.pointwise(x)  # Channel adjustment
        x = self.bn(x)         # Batch normalization
        x = self.relu(x)       # Activation function
        return x
        
class LightGCN(nn.Module):
    def __init__(self, in_channels):
        super(LightGCN, self).__init__()
        # Define a simple GCN layer
        self.gcn_layer = nn.Linear(in_channels, in_channels//2)
    
    def forward(self, x):
        # A simple forward pass through the GCN layer
        return self.gcn_layer(x)

      
class _SimpleSegmentationModel1(nn.Module):# 返回的12张 一个数组，
    # general segmentation model
    def __init__(self):
        super(_SimpleSegmentationModel1, self).__init__()
       # self.cl = nn.Conv2d(3, 8, 1)
       # self.vmunet = VSSM(in_chans=3,
        #                   num_classes=1,
       #                    depths=[2, 2, 2, 2],
       #                    depths_decoder=[2, 2, 2, 1],
       #                    drop_path_rate=0.2,
       #                 )
                        
        self.lcvmunet = CCViM(input_channels=3,num_classes=1,depths=[2, 2, 2, 2],
                                      decoder_depth=[2, 2, 2, 1],
                                      drop_path_rate=0.2)
        self.load_ckpt_path = "/lab/ly/YanLiu/BUsegNew/BUsegNew/lib/backbone/local_vssm_small.ckpt"
        if self.load_ckpt_path is not None:
            model_dict = self.lcvmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            if 'state_dict' in modelCheckpoint:
                # 如果存在 'state_dict' 键
                print(f'#####')
                pretrained_dict = modelCheckpoint['state_dict']
            if 'scheduler_state_dict' in modelCheckpoint:
                # 如果存在 'scheduler_state_dict' 键
                print(f'#####')
                pretrained_dict = modelCheckpoint['scheduler_state_dict']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.lcvmunet.load_state_dict(model_dict)  # 加载预训练参数

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.lcvmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            if 'state_dict' in modelCheckpoint:
                # 如果存在 'state_dict' 键
                print(f'#####')
                pretrained_odict = modelCheckpoint['state_dict']
            if 'scheduler_state_dict' in modelCheckpoint:
                # 如果存在 'scheduler_state_dict' 键
                print(f'#####')
                pretrained_odict = modelCheckpoint['scheduler_state_dict']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.lcvmunet.load_state_dict(model_dict)

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
   # def _dynamic_fusion(self, feat_a, feat_b, fusion_conv):
   #     spatial_weight = torch.sigmoid(fusion_conv(feat_a + feat_b))
    #    return spatial_weight * feat_a + (1 - spatial_weight) * feat_b  # 修正变量名

    def forward(self, x):
  
    #    plt.imshow(x[0][0].detach().cpu().numpy(), cmap='inferno')#plt.imshow(output_mapnm, cmap='inferno') plt.contourf  viridis
    #    plt.colorbar()
    #    plt.title('x Map')
    #    plt.show()    
        input_shape = x.shape[-2:]
     
        pixoutput = self.lcvmunet(x)
      #  pixoutput = F.interpolate(pixoutput, scale_factor=2, mode='bilinear')#[4,1,224，224]
       # f = self.dy_sample5(f)
     #   pixoutput = f
        
        return pixoutput        

           
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    model = _segm_pvtv3(1).cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    prediction1 = model(input_tensor)
