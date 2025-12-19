import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from .modules import BoundaryWiseAttentionGate2D, BoundaryWiseAttentionGateAtrous2D


class Transformer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=2,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):#x, None, self.query_embed.weight, pos_embed)
        bs, c, h, w = src.shape
       # src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        #print('query_embed=',query_embed)
       # print('tgt=',tgt)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #         print("Trans Encoder",memory.shape)
        
        #memory 是编码器的输出,会送到解码器去
        hs = self.decoder(tgt,
                          memory,
                          memory_key_padding_mask=mask,
                          pos=pos_embed,
                          query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class BoundaryAwareTransformer(nn.Module):
    def __init__(self,
                 point_pred_layers=2,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False,
                 return_intermediate_dec=False,
                # BAG_type='2D',
                 Atrous=False):
        super().__init__()
        self.num_decoder_layers = num_decoder_layers

      #  encoder_layer = BoundaryAwareTransformerEncoderLayer(
       #     d_model, nhead, BAG_type, Atrous, dim_feedforward, dropout,
     #       activation, normalize_before)
        encoder_layer = BoundaryAwareTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = BoundaryAwareTransformerEncoder(point_pred_layers,
                                                       encoder_layer,
                                                       num_encoder_layers,
                                                       encoder_norm)
        if num_decoder_layers > 0:
            decoder_layer = TransformerDecoderLayer(d_model, nhead,
                                                    dim_feedforward, dropout,
                                                    activation,
                                                    normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_decoder_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
       # print('query_embed',query_embed)
        bs, c, h, w = src.shape
       # print("w==============",w)
       # src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
       # print('query_embed=',query_embed)
      #  print('tgt=',tgt)
        memory = self.encoder(src,
                                    src_key_padding_mask=mask,
                                    pos=pos_embed,
                                    height=h,
                                    width=w)
        if self.num_decoder_layers > 0:
            hs = self.decoder(tgt,
                              memory,
                              memory_key_padding_mask=mask,
                              pos=pos_embed,
                              query_pos=query_embed)
            return hs.transpose(1, 2), memory.permute(1, 2,
                                                      0).view(bs, c, h,
                                                              w)
        else:
            return tgt.transpose(1, 2), memory.permute(1, 2,
                                                       0).view(bs, c, h,
                                                               w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class BoundaryAwareTransformerEncoder(nn.Module):
    def __init__(self,
                 point_pred_layers,
                 encoder_layer,
                 num_layers,
                 norm=None):
        super().__init__()
        self.point_pred_layers = point_pred_layers
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                height: int = 32,
                width: int = 32):
        output = src
     #   weights = []

        for layer_i, layer in enumerate(self.layers):
            output = layer(True,
                                   output,
                                   src_mask=mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   pos=pos,
                                   height=height,
                                   width=width)
         #   weights.append(weight)
        output = output.flatten(2).permute(2, 0, 1).contiguous()
        if self.norm is not None:
            output = self.norm(output)

       # return output, weights
        return output


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


# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3,  padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x
        
class TransformerEncoderLayer(nn.Module):
   # def __init__(self,dim, head_num):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_size = d_model
        self.head_num = nhead
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1,  padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1,  padding=0)  # qkv_v
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp(self.hidden_size)
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1,  padding=0)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(d_model)
        self.normalize_before = normalize_before
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
                     
        #b, c = src.shape[1:]
       # src = src.permute(1, 2, 0).view(b, c, 28, 28)
        h = src
        B, C, H, W = src.size()

        src = src.view(B, C, H*W).permute(0, 2, 1).contiguous()
        src = self.attention_norm(src).permute(0, 2, 1).contiguous()
        src = src.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(src), 2, dim=1)
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

        src = attn_out + h
        src = src.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = src
        src = self.ffn_norm(src)
        src = self.ffn(src)
        src = src + h
        src = src.permute(0, 2, 1).contiguous()
        src = src.view(B, C, H, W)

        src = self.PEG(src)
       # src = src.flatten(2).permute(2, 0, 1).contiguous()

        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
       # print("---------",src.size())
       # b, c = src.shape[1:]
       # src = src.permute(1, 2, 0).view(b, c, 28, 28)
        h = src
       # print('====================',src.size())
        B, C, H, W = src.size()

        src = src.view(B, C, H*W).permute(0, 2, 1).contiguous()
        src = self.attention_norm(src).permute(0, 2, 1).contiguous()
        src = src.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(src), 2, dim=1)
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

        src = attn_out + h
        src = src.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = src
        src = self.ffn_norm(src)
        src = self.ffn(src)
        src = src + h
        src = src.permute(0, 2, 1).contiguous()
        src = src.view(B, C, H, W)

        src = self.PEG(src)
       # src = src.flatten(2).permute(2, 0, 1).contiguous()
       # print('++++++++++++++++++++++++++++',src.size())
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
   
        
class TransformerEncoderLayer1(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)#1升高
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)#2降低

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q,
                              k,
                              value=src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0] #MSA(input)
        src = src + self.dropout1(src2)#MSA(input)+dropout(input)
        src = self.norm1(src)# norm(MSA(input)+dropout(input))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))#MLP
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q,
                              k,
                              value=src2,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

        
class BoundaryAwareTransformerEncoderLayer(TransformerEncoderLayer):
    "    Add Boundary-wise Attention Gate to Transformer's Encoder"

    def __init__(self,
                 d_model,
                 nhead,
               #  BAG_type='2D',
             #    Atrous=True,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         normalize_before)
       # if BAG_type == '2D':
      #      if Atrous:
      #          self.BAG = BoundaryWiseAttentionGateAtrous2D(d_model)
      
        self.BAG = BoundaryWiseAttentionGateAtrous2D(128)
      #  self.BAG_type = BAG_type

    def forward(self,
                use_bag,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                height: int = 32,
                width: int = 32):
        if self.normalize_before:
            features = self.forward_pre(src, src_mask, src_key_padding_mask, pos)
           # b, c = features.shape[1:]
            #features = features.permute(1, 2, 0).view(b, c, height, width)
            features = self.BAG(features)
          #  features = features.flatten(2).permute(2, 0, 1).contiguous()
            return features
        features = self.forward_post(src, src_mask, src_key_padding_mask, pos)
        
      #  b, c = features.shape[1:]
       # features = features.permute(1, 2, 0).view(b, c, height, width)
        features = self.BAG(features)
       # features = features.flatten(2).permute(2, 0, 1).contiguous()
        return features


class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output,
                           memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)#128  2048
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)#  2048  128

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),#CrossAttentionLayer
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)#CrossAttentionLayer
        tgt = self.norm2(tgt)#CrossAttentionLayer             
        q = k = self.with_pos_embed(tgt, query_pos)#MSA(tgt)  self SelfAttentionLayer
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0] #MSA(tgt)  self SelfAttentionLayer
        tgt = tgt + self.dropout1(tgt2)#  dropout(MSA(tgt))+tgt  self SelfAttentionLayer
        tgt = self.norm1(tgt)#    norm(dropoutMSA(tgt))+tgt)     self SelfAttentionLayer
    
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) #FFN """ Very simple multi-layer perceptron (also called FFN)"""
        tgt = tgt + self.dropout3(tgt2)#FFN """ Very simple multi-layer perceptron (also called FFN)"""
        tgt = self.norm3(tgt)#FFN  """ Very simple multi-layer perceptron (also called FFN)"""
        return tgt

    def forward_pre(self,
                    tgt,
                    memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "leaky relu":
        return F.leaky_relu
    if activation == "selu":
        return F.selu
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        F"activation should be relu, gelu, glu, leaky relu or selu, not {activation}."
    )
