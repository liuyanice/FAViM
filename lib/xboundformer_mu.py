from turtle import forward
import torch
import cv2
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from lib.Position_embedding import PositionEmbeddingLearned
from lib.modules import xboundlearner, xboundlearnerv2, _simple_learner, xboundlearnerv5
from lib.vision_transformers import in_scale_transformer
from lib.sampling_points import sampling_points, point_sample
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
#from lib.pvtv2 import pvt_v2_b2  #
from lib.pvtv2 import pvt_v2_b1
#from lib.segformer import mit_b1
#from lib.pvt import pvt_tiny  #
from os.path import join as pjoin
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from torch.nn.modules.utils import _pair
from .memory import XBM1, XBM2, XBM3, XBM4
from torch.distributions import Normal
from .seed_init import place_seed_points
#from mamba_ssm import Mamba
from .vit_seg_modeling_resnet_skip import ResNetV2
import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
#from efficientnet_pytorch import EfficientNet

   
def _segm_pvtv2(num_classes, im_num, ex_num, xbound, trainsize): #(1,1,1,1,[224,224])
   # backbone = pvt_v2_b2(img_size=trainsize)
    #backbone = pvt_tiny(img_size=trainsize)
   # backbone = pvt_v2_b1(img_size=trainsize)
  #  backbone = mit_b1(img_size=trainsize)
   # layer_name = 'layer2'
    backbone = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1,
                                embed_dim=96,
                                #depths=[2,2,6,2],
                                depths=[2,2,2,2],
                                decoder_depths=[ 2, 2, 2, 1],
                                num_heads=[3,6,12,24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                final_upsample="expand_first",
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
  #  backbone1 = EfficientNet.from_pretrained('efficientnet-b2')
    pretrained_path = r'/lab/ly/YanLiu/BUsegNew/BUsegNew/lib/backbone/swin_tiny_patch4_window7_224.pth'
    if pretrained_path is not None:
    
    #    path = r'/lab/ly/YanLiu/BUsegNew/BUsegNew/lib/backbone/pvt_v2_b2.pth'
    #    path = r'/lab/ly/YanLiu/BUsegNew/BUsegNew/lib/backbone/pvt_tiny.pth'
       # path = r'/home/liuyan/TestMBUSeg/lib/backbone/swin_tiny_patch4_window7_224.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      #  path = r'/lab/ly/YanLiu/BUsegNew/BUsegNew/lib/backbone/mit_b1.pth'
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                backbone.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")
        model_dict = backbone.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]
      #  model_dict.update(state_dict)
        backbone.load_state_dict(full_dict, strict=False)
    #定义线性分类器
    classifier = _simple_classifier(num_classes)
    #加载pixelDecoderModel
    pixelDecoderModel1 = pixelDecoderModel()
    PointHead1 = PointHead()
    #加载backbone完毕，调用自己的分割网络_SimpleSegmentationModel.
    model = _SimpleSegmentationModel(backbone, classifier, im_num, ex_num,
                                     xbound, pixelDecoderModel1, PointHead1)
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
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out
        

class PVMLayer3(nn.Module):
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
        self.up = nn.Upsample(scale_factor=2)
        self.se = SE(512)
       # self.ca = ChannelAttention(320)
      #  self.ca1 = ChannelAttention(512)
      #  self.conv = Conv2dIBNormRelu(512, 320, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)
     #   self.conv2 = Conv2dIBNormRelu(512, 320, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(640, 320, 1, 1)
        self.conv2 = nn.Conv2d(512, 320, 1, 1)
        self.qkv_local_fc1 = Conv2dIBNormRelu(320, 320//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = Conv2dIBNormRelu(320//4, 320, 1, 1)
        self.depth_wise = Conv2dIBNormRelu(320//4, 320//4, kernel_size=3, padding=1, groups=320//4)
      #  self.relu = nn.ReLU()
        
    def forward(self,f, x):
        c = self.up(x)#512
        z = self.conv2(c)
        
     #   global_features = torch.cat([c, f], dim=1)#512+320=832
        c = self.se(c)    #512   
        if c.dtype == torch.float16:
            c = c.type(torch.float32)
        B, C = c.shape[:2]
        assert C == self.input_dim
        n_tokens = c.shape[2:].numel()
        img_dims = c.shape[2:]
        x_flat = c.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)#320
     #   combined_features = torch.cat([out, HF, LF], dim=1)#320 320 512
    #    out = self.conv1(combined_features)
       # X_1 = out + self.ca(out)
        X_1 = out * f + out#320
        
        global_features = torch.cat([z, f], dim=1)#320+320=640
     #   X_1 = self.up(out + z)+f
       # X_2 = global_features
        X_2 = self.conv1(global_features)#320
     #   X_1 = X_1.view(B_c, C_c, H_c*W_c).permute(0, 2, 1).contiguous()
  #      print("X_2===",X_1.shape)
        X_2 = self.qkv_local_fc1(X_2)
  #      print("X_2===",X_1.shape)
        X_2 = self.depth_wise(X_2)
        X_2 = self.qkv_local_fc2(X_2)
  #      print("X_1===",X_1.shape)
  #      print("X_2===",X_2.shape)
        X_1_saf = X_1 + X_2
   #     out = (X_1_saf*self.up(c_c))*f#+self.up(c_c)#+self.up(x)
       # up = self.up(X_1_saf)
       # out = out+f
        out = X_1_saf#*f
      #  combined_features = torch.cat([up, f], dim=1)
      #  out = self.conv1(combined_features)
        return out
        
class PVMLayer2(nn.Module):
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
        self.up = nn.Upsample(scale_factor=2)
        self.se = SE(320)
  #      self.ca = ChannelAttention(128)
   #     self.conv = Conv2dIBNormRelu(320, out_channels=128, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(256, 128, 1, 1)
        self.conv2 = nn.Conv2d(320, 128, 1, 1)
        self.qkv_local_fc1 = Conv2dIBNormRelu(128, 128//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = Conv2dIBNormRelu(128//4, 128, 1, 1)
        self.depth_wise = Conv2dIBNormRelu(128//4, 128//4, kernel_size=3, padding=1, groups=128//4)
       # self.relu = nn.ReLU()
    def forward(self,f, x):#看了f的热力图非常好
        c = self.up(x)
        z = self.conv2(c)
        
     #   global_features = torch.cat([c, f], dim=1)#512+320=832
        c = self.se(c)       
        if c.dtype == torch.float16:
            c = c.type(torch.float32)
        B, C = c.shape[:2]
        assert C == self.input_dim
        n_tokens = c.shape[2:].numel()
        img_dims = c.shape[2:]
        x_flat = c.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)#320
     #   combined_features = torch.cat([out, HF, LF], dim=1)#320 320 512
    #    out = self.conv1(combined_features)
       # X_1 = out + self.ca(out)
        X_1 = out * f + out#320
        
        global_features = torch.cat([z, f], dim=1)#320+320=640  可以用out
     #   X_1 = self.up(out + z)+f
       # X_2 = global_features
        X_2 = self.conv1(global_features)
     #   X_1 = X_1.view(B_c, C_c, H_c*W_c).permute(0, 2, 1).contiguous()
  #      print("X_2===",X_1.shape)
        X_2 = self.qkv_local_fc1(X_2)
  #      print("X_2===",X_1.shape)
        X_2 = self.depth_wise(X_2)
        X_2 = self.qkv_local_fc2(X_2)
  #      print("X_1===",X_1.shape)
  #      print("X_2===",X_2.shape)
        X_1_saf = X_1 + X_2
   #     out = (X_1_saf*self.up(c_c))*f#+self.up(c_c)#+self.up(x)
       # up = self.up(X_1_saf)
       # out = out+f
        out = X_1_saf#*f
      #  combined_features = torch.cat([up, f], dim=1)
      #  out = self.conv1(combined_features)
        return out

class PVMLayer1(nn.Module):
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
        self.up = nn.Upsample(scale_factor=2)
        self.se = SE(128)
    #    self.ca = ChannelAttention(64)
    #    self.conv = Conv2dIBNormRelu(128, out_channels=64, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(128, 64, 1, 1)
        self.conv2 = nn.Conv2d(128, 64, 1, 1)
        self.qkv_local_fc1 = Conv2dIBNormRelu(64, 64//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = Conv2dIBNormRelu(64//4, 64, 1, 1)
        self.depth_wise = Conv2dIBNormRelu(64//4, 64//4, kernel_size=3, padding=1, groups=64//4)
       # self.relu = nn.ReLU()
    def forward(self,f, x):
        c = self.up(x)
        z = self.conv2(c)
        
     #   global_features = torch.cat([c, f], dim=1)#512+320=832
        c = self.se(c)       
        if c.dtype == torch.float16:
            c = c.type(torch.float32)
        B, C = c.shape[:2]
        assert C == self.input_dim
        n_tokens = c.shape[2:].numel()
        img_dims = c.shape[2:]
        x_flat = c.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)#320
     #   combined_features = torch.cat([out, HF, LF], dim=1)#320 320 512
    #    out = self.conv1(combined_features)
       # X_1 = out + self.ca(out)
        X_1 = out * f + out#320
     #   print("z==", z.shape)
    #    print("f==", f.shape)
        global_features = torch.cat([z, f], dim=1)#320+320=640
     #   X_1 = self.up(out + z)+f
       # X_2 = global_features
        X_2 = self.conv1(global_features)
     #   X_1 = X_1.view(B_c, C_c, H_c*W_c).permute(0, 2, 1).contiguous()
  #      print("X_2===",X_1.shape)
        X_2 = self.qkv_local_fc1(X_2)
  #      print("X_2===",X_1.shape)
        X_2 = self.depth_wise(X_2)
        X_2 = self.qkv_local_fc2(X_2)
  #      print("X_1===",X_1.shape)
  #      print("X_2===",X_2.shape)
        X_1_saf = X_1 + X_2
   #     out = (X_1_saf*self.up(c_c))*f#+self.up(c_c)#+self.up(x)
       # up = self.up(X_1_saf)
       # out = out+f
        out = X_1_saf#*f
      #  combined_features = torch.cat([up, f], dim=1)
      #  out = self.conv1(combined_features)
        return out
class CADM3(nn.Module):
    def __init__(self, dim, head_num=8):
        super(CADM3, self).__init__()
        self.hidden_size = dim 
        self.head_num = head_num
        self.conv = nn.Conv2d(dim, out_channels=320, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
      #  self.attention_norm = nn.LayerNorm(dim)
      #  self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_f = nn.Linear(576, 144)  # qkv_h 484  121(352) 576   144(384)
        self.qkv_local_c = nn.Linear(144, 144)  # qkv_v 121  121      144  144
        self.qkv_local_fc1 = nn.Conv2d(320, 320//4, 1, 1) #nn.Linear(320, 320*4)  # qkv_h
        self.qkv_local_fc2 = nn.Conv2d(320//4, 320, 1, 1)# nn.Linear(320*4, 320)  # qkv_v
    #    self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
       # self.ffn_norm = nn.LayerNorm(dim)
      #  self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
      #  self.PEG = PEG(dim)
        self.ca = ChannelAttention(320)
        self.depth_wise = nn.Conv2d(320//4, 320//4, kernel_size=3, padding=1, groups=320//4)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        
        
    def forward(self, f, c):
  #      print("f.shape===",f.shape)
  #      print("c.shape===",c.shape)
        c = self.conv(c)
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
        x = self.up(X_1_saf *c_c)*f
  #      print("x.shape===",x.shape)
        return x 



class CADM2(nn.Module):
    def __init__(self, dim, head_num=8):
        super(CADM2, self).__init__()
        self.hidden_size = dim 
        self.head_num = head_num
        self.conv = nn.Conv2d(dim, out_channels=128, kernel_size=3,  stride=1, dilation=1, padding=1, bias=False)#groups=1,
      #  self.attention_norm = nn.LayerNorm(dim)
      #  self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_f = nn.Linear(2304, 576)  # qkv_h 1936    484(352) 2304   576(384)
        self.qkv_local_c = nn.Linear(576, 576)  # qkv_v  484     484      576    576
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
        
        
    def forward(self, f, c):
    #    print("f.shape===",f.shape)
    #    print("c.shape===",c.shape)
        c = self.conv(c)
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
        
        
        X_1 = X_1.view(B_c, C_c, H_c, W_c)
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
  #      print("c_c.shape===",c_c.shape)
   #     print("X_1_saf.shape===",X_1_saf.shape)
  #      print("f.shape===",f.shape)
      #  X_1 = X_1.permute(0, 2, 1).contiguous()
      #  X_O = X_1.view(B_c, C_c, H_c, W_c)
        x = self.up(X_1_saf *c_c)*f
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
        self.qkv_local_f = nn.Linear(9216, 2304)  # qkv_h 7744   1936(352)     9216     2304    (384)
        self.qkv_local_c = nn.Linear(2304, 2304)  # qkv_v 1936   1936          2304     2304
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
        
        
    def forward(self, f, c):
    #    print("f.shape===",f.shape)
    #    print("c.shape===",c.shape)
        c = self.conv(c)
        c_c = self.ca(c)
    #    print("c_c.shape===",c_c.shape)
        h_f = f
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
        X_1= X_1.permute(0, 2, 1).contiguous()
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
    #    print("c_c.shape===",c_c.shape)
   #     print("X_1_saf.shape===",X_1_saf.shape)
   #     print("f.shape===",f.shape)
      #  X_1 = X_1.permute(0, 2, 1).contiguous()
      #  X_O = X_1.view(B_c, C_c, H_c, W_c)
        x = self.up(X_1_saf *c_c)*f
    #    print("x.shape===",x.shape)
        return x 
        
        
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
        return self.sigmoid(out)*x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        
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


class ChannelBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2*out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        
        self.conv4 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        conv1_out = self.conv1(x)#out_channels
        batch1_out = self.batch1(conv1_out)
        relu1_out = self.relu1(batch1_out)#out_channels
        
        conv2_out = self.conv2(x)#out_channels
        batch2_out = self.batch2(conv2_out)
        relu2_out = self.relu2(batch2_out)#out_channels
        
        concat_out = torch.cat((relu1_out, relu2_out), dim=1)#2*out_channels
    #    print("concat_out===", concat_out.shape)
        global_pool_out = self.global_pool(concat_out).squeeze()#2*out_channels
    #    print("global_pool_out===", global_pool_out.shape)
        if self.training:
          fc1_out = self.fc1(global_pool_out)
        else:
          fc1_out = self.fc1(global_pool_out.unsqueeze(0))
    #    print("fc1_out===", fc1_out.shape)
        bn1_out = self.bn1(fc1_out)#out_channels
        relu3_out = self.relu3(bn1_out)
        fc2_out = self.fc2(relu3_out)
   #     print("fc2_out===", fc2_out.shape)
        sigmoid_out = self.sigmoid(fc2_out)#out_channels
   #     print("sigmoid_out===", sigmoid_out.shape)
        a = sigmoid_out.view(-1, fc2_out.size(1), 1, 1)
        a1 = 1 - sigmoid_out
        a1 = a1.view(-1, fc2_out.size(1), 1, 1)
        
        y = torch.mul(relu1_out, a)#out_channels
        y1 = torch.mul(relu2_out, a1)#out_channels
        
        data_a_a1 = torch.cat((y, y1), dim=1)#2*out_channels
   #     print("data_a_a1===",data_a_a1.shape)
        conv3_out = self.conv4(data_a_a1)
        batch3_out = self.bn4(conv3_out)
        relu3_out = self.relu4(batch3_out)##out_channels
        
        return relu3_out

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.batch1 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, data, channel_data):#in_channels    #out_channels
        conv1_out = self.conv1(data)#in_channels
        batch1_out = self.batch1(conv1_out)
        relu1_out = self.relu1(batch1_out)#out_channels
        
        conv2_out = self.conv2(relu1_out)
        batch2_out = self.batch2(conv2_out)
        relu2_out = self.relu2(batch2_out)
        spatial_data = relu2_out#out_channels
        
        data3 = channel_data + spatial_data#out_channels
        data3 = F.relu(data3)
        data3 = self.conv3(data3)#
        data3 = torch.sigmoid(data3)#out_channels
        
      #  a = expend_as(data3, filte)  # Need to implement expend_as function
        y = torch.mul(data3, channel_data)#out_channels
        
        a1 = 1 - data3
    #    a1 = expend_as(a1, filte)  # Need to implement expend_as function
        y1 = torch.mul(a1, spatial_data)
        
        data_a_a1 = torch.cat((y, y1), dim=1)
      #  print("data_a_a1=====",data_a_a1.shape)
        conv3_out = self.conv4(data_a_a1)
        batch3_out = self.bn4(conv3_out)
        
        return batch3_out
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.chan = ChannelBlock(in_channels, out_channels)
        self.spat = SpatialBlock(in_channels, out_channels)

    def forward(self, x):
        y = self.chan(x)
        
        x = self.spat(x,y)
        return x
        

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, data):
        conv1_out = self.conv1(data)
        batch1_out = self.batch1(conv1_out)
        leakyrelu1_out = self.leakyrelu1(batch1_out)

        conv2_out = self.conv2(leakyrelu1_out)
        batch2_out = self.batch2(conv2_out)
        leakyrelu2_out = self.leakyrelu2(batch2_out)

        return leakyrelu2_out

class UpData1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpData1, self).__init__()
        self.upsample = nn.Upsample()
        #self.concatenate = nn.Concat(dim=1)
        self.convblock = ConvBlock(in_channels, out_channels)

    def forward(self, data, skipdata):
        shape = skipdata.size()
        shape1 = data.size()

        data1 = F.interpolate(data, size=(shape[2], shape[3]), mode='nearest')
        concatenated = torch.cat((skipdata, data1), dim=1)
        concatenated = self.convblock(concatenated)

        return concatenated



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
      #  self.config = config
        head_channels = 512
        hidden_size =768
        self.conv_more = Conv2dReLU(
            hidden_size, #config.hidden_size
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = [256,128,64,16] #config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        #if self.config.n_skip != 0:
        skip_channels = [512,256,64,16]#self.config.skip_channels
        for i in range(1):  # re-select the skip channels according to n_skip
            skip_channels[3-i]=0

    #    else:
     #       skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < 3) else None  #self.config.n_skip
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, in_channels=3):
        super(Embeddings, self).__init__()
      #  self.hybrid = None
     #   self.config = config
        self.hybrid = True
        img_size = _pair(img_size)

   #     if config.patches.get("grid") is not None:   # ResNet
        grid_size = (16, 16)
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
        
    #    else:
    #        patch_size = _pair(config.patches["size"])
    #        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
    #        self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=(3,4,9), width_factor=1)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=768,#config.hidden_size
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = Dropout(0.1)#config.transformer["dropout_rate"]


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features



class Trans_Attention(nn.Module):
    def __init__(self, vis):
        super(Trans_Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)

        self.out = Linear(768, 768)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
        
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = Mlp(hidden_size=768)
        self.attn = Trans_Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            
class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)#config.hidden_size
        for _ in range(12):#config.transformer["num_layers"]
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
        
class Transformer(nn.Module):
    def __init__(self, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features
        
class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super(SegmentationHead, self).__init__()#conv2d, upsampling
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.upsampling(x)
        return x
    
class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1,
                                embed_dim=96,                                                                                                      
                                depths=[2,2,6,2],
                                num_heads=[3,6,12,24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
            
class _SimpleSegmentationModel(nn.Module):# 返回的12张 一个数组，
    # general segmentation model
    def __init__(self, backbone, classifier, im_num, ex_num, xbound, pixelDecoderModel, PointHead):
        super(_SimpleSegmentationModel, self).__init__()
     #   resnet34 = torchvision.models.resnet34(pretrained=True)
     #   filters = [64, 128, 256, 512]
     #   self.firstlayer = nn.Sequential(*list(resnet34.children())[:3])
     #   self.maxpool = list(resnet34.children())[3]
     #   self.encoder1 = resnet34.layer1
     #   self.encoder2 = resnet34.layer2
     #   self.encoder3 = resnet34.layer3
     #   self.encoder4 = resnet34.layer4
        #self.n_channels = 3
       # self.n_classes = 1
        self.transformer = backbone
        #self.transformer = SwinTransformerSys(img_size=224,
       #                         patch_size=4,
      #                          in_chans=3,
       #                         num_classes=1,
       #                         embed_dim=96,
       ##                         depths=[2,2,6,2],
       #                         num_heads=[3,6,12,24],
        #                        window_size=7,
        #                        mlp_ratio=4.,
        #                        qkv_bias=True,
        #                        qk_scale=None,
        #                        drop_rate=0.0,
        #                        drop_path_rate=0.1,
        #                        ape=False,
        #                        patch_norm=True,
        #                        use_checkpoint=False)
    #    self.transformer = SwinUnet(img_size=384, num_classes=1).cuda()
       # self.transformer = Transformer(img_size=384, vis=False)
     #   self.decoder = DecoderCup()
     #   self.segmentation_head = SegmentationHead(
     #       in_channels=16,
     #       out_channels=1,
     #       kernel_size=3,
     #   )
     #   self.outc= nn.Conv2d(64, 1, kernel_size=1, padding=0)
    def forward(self, x, y):
    #    print("up7=======", up7.shape)#4 64 384 384
        
        logits = self.transformer(x)  # (B, n_patch, hidden)
   #     print("x=====",x.shape)#4, 567,768
   #     print("features[0]=====",features[0].shape)#4  512  48 48 
    #    print("features[1]=====",features[1].shape)#4  256  96 96
   #     print("features[2]=====",features[2].shape)#4  64  192 192
     #   x = self.decoder(x, features)
      #  logits = self.segmentation_head(x)
       # return logits
      #  pixoutput = self.outc(logits)
        pixoutput = logits 
      #  pixoutput = self.outc(up7)
     #   return pixoutput
        
        outputs = [
                
            ]
        
        
                                     
     #   print('outputs=============',outputs[0].shape)#[12, 1, 224, 224]
      #  print('pixoutput=======',pixoutput.shape)
        return outputs, pixoutput #, rend, points
           



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    model = _segm_pvtv2(1).cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    prediction1 = model(input_tensor)
