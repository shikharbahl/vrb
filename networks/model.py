from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

import os
import logging
import math
import numpy as np
from torch import nn, Tensor
from einops import rearrange
from networks.layer import EncoderBlock
from networks.net_utils import trunc_normal_, get_pad_mask, get_subsequent_mask, traj_affordance_dist
from torch.nn import init
from torch.nn import functional as F
import torchvision
from torch.nn.modules.transformer import MultiheadAttention
from collections import OrderedDict
import copy
from typing import Optional, List
from torchvision.models import resnet18, resnet50
BN_MOMENTUM = 0.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.net_utils import DropPath, get_pad_mask
from einops import rearrange




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class Encoder_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(Encoder_PositionalEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        B, T = x.shape[:2]
        if T != self.position_embedding.size(1):
            position_embedding = self.position_embedding.transpose(1, 2)
            new_position_embedding = F.interpolate(position_embedding, size=(T), mode='nearest')
            new_position_embedding = new_position_embedding.transpose(1, 2)
            x = x + new_position_embedding
        else:
            x = x + self.position_embedding
        return x


class Encoder(nn.Module):
    def __init__(self, num_patches=5, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 dropout=0., time_embed_type=None, num_frames=None):
        super().__init__()
        if time_embed_type is None or num_frames is None:
            time_embed_type = 'sin'
        self.time_embed_type = time_embed_type
        self.num_patches = num_patches  # (hand, object global feature patches, default: 5)
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Encoder_PositionalEmbedding(embed_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(embed_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.encoder_blocks = nn.ModuleList([EncoderBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed'}

    def forward(self, x, mask=None):
        # import ipdb; ipdb.set_trace()
        B, T, N = x.shape[:3]

        x = rearrange(x, 'b t n m -> (b t) n m', b=B, t=T, n=N)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = self.time_embed(x)
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        mask = mask.transpose(1, 2)
        for blk in self.encoder_blocks:
            x = blk(x, B, T, N, mask=mask)

        x = rearrange(x, 'b (n t) m -> b t n m', b=B, t=T, n=N)
        x = self.norm(x)
        return x




def conv_block(in_channels, out_channels, kernel_size, stride, dilation=1, bias=True, batch_norm=True, layer_norm=False, activation='ReLU'):
    padding = (dilation*(kernel_size-1)+2-stride)//2
    seq_modules = nn.Sequential()
    seq_modules.add_module('conv', \
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias))
    if batch_norm:
        seq_modules.add_module('norm', nn.BatchNorm2d(out_channels))
    elif layer_norm:
        seq_modules.add_module('norm', LayerNorm())
    if activation is not None:
        seq_modules.add_module('relu', getattr(nn, activation)(inplace=True))
    return seq_modules

def get_coord(x, other_axis, axis_size):
    "get x-y coordinates"
    g_c_prob = torch.mean(x, dim=other_axis)  # B,NMAP,W
    g_c_prob = F.softmax(g_c_prob, dim=2) # B,NMAP,W
    coord_pt = torch.linspace(0, 1.0, axis_size).to(x.device) # W
    coord_pt = coord_pt.view(1, 1, axis_size) # 1,1,W
    g_c = torch.sum(g_c_prob * coord_pt, dim=2) # B,NMAP
    return g_c, g_c_prob

def positional_encoding(d_model, H, W): 
    max_len = H*W
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe


class ImageEncoder(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = nn.Sigmoid()
        self.normlayer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
        self.convnet = resnet
        self.convnet.fc = nn.Identity()


    def forward(self, obs, num_ims = 1, obs_shape = [3, 224, 224]):
        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                        torchvision.transforms.Resize(256),
                        torchvision.transforms.CenterCrop(224),
                        self.normlayer,
                )
        else:
            preprocess = nn.Sequential(
                        self.normlayer,
                )
        obs = obs.float() /  255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h


class VRBModel(nn.Module):
    
    def __init__(self, src_in_features, 
                 num_patches=1,hand_head=None,
                 embed_dim=512, coord_dim=64,  hidden_dim=128, num_heads=8, enc_depth=6,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.,
                 encoder_time_embed_type='sin',  freeze_rep=False,
                 num_frames_input=None, n_maps=5, resnet_type='resnet18', 
                 var=False, attn_kp=True, attn_kp_fc=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.coord_dim = coord_dim
        self.downproject = nn.Linear(src_in_features, embed_dim)
        self.n_maps = n_maps
        
        self.encoder = Encoder(num_patches=num_patches,
                               embed_dim=embed_dim, depth=enc_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate, norm_layer=norm_layer, dropout=dropout,
                               time_embed_type=encoder_time_embed_type, num_frames=num_frames_input)
        
        self.var = var
        
        if resnet_type == 'resnet18': 
            self.deconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.deconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.deconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.cb1 = conv_block(64, 10, kernel_size=3, stride=1)
            self.cb2 = conv_block(10, 5, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm2d(256).cuda()
            self.bn2 = nn.BatchNorm2d(128).cuda()
            self.bn3 = nn.BatchNorm2d(64).cuda()
            r3m = resnet18()
        else: 
            self.deconv1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
            self.deconv2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.deconv3 = nn.ConvTranspose2d(512, 128, 2, stride=4)
            self.cb1 = conv_block(128, 10, kernel_size=3, stride=1)
            self.cb2 = conv_block(10, 5, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm2d(1024)
            self.bn2 = nn.BatchNorm2d(512)
            self.bn3 = nn.BatchNorm2d(128)            
            r3m = resnet50()
            
        self.r3m = ImageEncoder(r3m)
        self.r3m.eval()

        if freeze_rep:
            for param in self.r3m.parameters(): 
                param.requires_grad = False
            
            
        
        self.var_MLP = nn.Sequential(
            nn.Linear(src_in_features, hidden_dim),
            nn.ELU(), 
            nn.Linear(hidden_dim, n_maps*2),)
        self.mean_lmbda = 5
        
        self.attn_kp = attn_kp
        self.attn_kp_fc = attn_kp_fc
        self.flatten = nn.Flatten(start_dim=2)
        self.self_attention = MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.1)
        self.fc_mu = nn.Linear(64, self.n_maps * 2)
        # Add positional encoding
        self.register_buffer("pos_encoding", positional_encoding(64, 110, 110))  # Change 8x8 to the expected spatial dimensions of h

        if self.attn_kp: 
            if resnet_type == 'resnet18': 
                self.cb1 = conv_block(64, 64, kernel_size=3, stride=1)
            else: 
                self.cb1 = conv_block(128, 64, kernel_size=3, stride=1)
            self.cb2 = conv_block(64, 64, kernel_size=3, stride=1)
            self.cb3 = conv_block(64, 5, kernel_size=1, stride=1)

        self.hand_head = hand_head
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encoder_input(self, feat):
        B, T = feat.shape[0], feat.shape[2]
        feat = self.downproject(feat)
        feat = feat.transpose(1, 2)
        return feat

    def r3m_forward(self, img): 
        preprocess = nn.Sequential(nn.Identity())
        obs = img.float()
        obs_p = preprocess(obs)
        h = self.r3m.convnet(obs_p)
        return h
    
    
    def encode_img(self, img): 
        feat = self.r3m_forward(img)
        if self.var: 
            log_cvar = self.var_MLP(feat).reshape(-1, self.n_maps, 2)
            cvar = torch.exp(log_cvar)
        else: 
            cvar = 0
        x = self.r3m.convnet.conv1(img)
        x = self.r3m.convnet.bn1(x)
        x = self.r3m.convnet.relu(x)
        x = self.r3m.convnet.maxpool(x)
        conv1 = self.r3m.convnet.layer1(x)
        conv2 = self.r3m.convnet.layer2(conv1)
        conv3 = self.r3m.convnet.layer3(conv2)
        conv4 = self.r3m.convnet.layer4(conv3)
        h = self.deconv1(conv4)
        h = self.bn1(h)
        h = self.r3m.convnet.layer1[0].relu(h)
        h = self.deconv2(h + conv3)
        h = self.bn2(h)
        h = self.r3m.convnet.layer1[0].relu(h)
        h = self.deconv3(h + conv2)
        h = self.bn3(h)
        h = self.r3m.convnet.layer1[0].relu(h)
        x = self.cb2(self.cb1(h))
        n = x.shape[2]
        if self.attn_kp: 
            B, C, H, W = x.shape
            h_flattened = self.flatten(x).transpose(1, 2)  # (B, H*W, C)
            # Add positional encoding
            h_flattened = h_flattened + self.pos_encoding.transpose(0, 1)[:, :H * W].to(h.device)
            attn_output, _ = self.self_attention(h_flattened, h_flattened, h_flattened)  # (B, H*W, C)
            attn_output = attn_output.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        if self.attn_kp and self.attn_kp_fc: 
            mu = self.fc_mu(F.adaptive_avg_pool2d(attn_output, (1, 1)).view(-1, C)).view(B, self.n_maps, 2)  # (B, n_maps, 2)
            return feat, mu, cvar
        elif self.attn_kp: 
            x = self.cb3(attn_output)
        cy, _  = get_coord(x, 3, n)
        cx, _ = get_coord(x, 2, n)
        mu = torch.stack([cy, cx], dim=2)
        return feat, mu, cvar

    def forward(self, feat, bbox_feat, valid_mask, future_hands, contact_point, future_valid):
        B = feat.shape[0]
        fe, mu, cvar = self.encode_img(feat)
        gt_mu, gt_cvar = contact_point[:, :self.n_maps], contact_point[:, self.n_maps:]
        if self.var: 
            obj_loss = self.mean_lmbda*torch.mean((mu - gt_mu)**2) + torch.mean((cvar - gt_cvar)**2)
        else: 
            obj_loss = self.mean_lmbda*torch.mean((mu - gt_mu)**2)
        fe = fe.reshape(B, 1, 1, -1)
        valid_mask = torch.ones_like(fe[:, :, :, -1]).to(feat.device)
        src_mask = valid_mask
        # downproject for transformer 
        fe = self.encoder_input(fe)
        # encode with transformer 
        x = self.encoder(fe, mask=src_mask)
        memory = x[:, -1, :, :]
        target_hand = future_hands
        # hand cvae
        pred_hand, traj_loss, traj_kl_loss = self.hand_head(memory[:, 0, :], target_hand, future_valid, contact_point=None, return_pred=True)
        obj_kl_loss = None
        return traj_loss, traj_kl_loss, obj_loss, obj_kl_loss

    def inference(self, feat, bbox_feat, valid_mask, future_valid=None, pred_len=4):
        B = feat.shape[0]
        fe, mu, cvar = self.encode_img(feat)
        if self.var: 
            pred_contact = torch.stack([mu, cvar], dim=1)
        else:
            pred_contact = torch.stack([mu, mu], dim=1)
         
        fe = fe.reshape(B, 1, 1, -1)
        valid_mask = torch.ones_like(fe[:, :, :, -1]).to(feat.device)
        src_mask = valid_mask
        
        fe = self.encoder_input(fe)
        x = self.encoder(fe, mask=src_mask)
        memory = x[:, -1, :, :]

        future_hand = self.hand_head.inference(memory[:, 0, :])
        future_hand = future_hand.reshape(B, -1, 2)
        return future_hand, pred_contact



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask,  pos=pos,
                               src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")








