# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import bias

import numpy as np
# from pyrsistent import v
import torch
import torch.nn as nn
#from .DCNv2.dcn_v2 import DCN

from .DCNv2.dcn import DeformableConv2d

BN_MOMENTUM = 0.1

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class transposedconvolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride = 1, with_bn=True):
        super(transposedconvolution, self).__init__()
        self.with_bn = with_bn
        self.tconv = nn.ConvTranspose2d(inp_dim, out_dim, k, stride=stride, bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.tconv(x)
        x   = self.bn(x)
        relu = self.relu(x)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, feature_maps=None):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return mySequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

class MergeUp(nn.Module):
    def forward(self, up1, up2, feature_maps=None):
        if feature_maps:
            tmp = up1 + up2
            for i in feature_maps:
                if up1.shape[-1] == i.shape[-1]:
                    tmp += i
            return tmp
        return up1 + up2

class MergeUp2(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

# class MergeUp3(nn.Module):
#     def __init__(self, inp_dim):
#         super(MergeUp3, self).__init__()
#         self.conv = convolution(3, inp_dim*2, inp_dim, stride=1)

#     def forward(self, up1, up2, feature_maps=None):
#         tmp = up1 + up2
#         if feature_maps:
#             for i in feature_maps:
#                 if up1.shape[-1] == i.shape[-1]:
#                     concat = torch.cat([tmp, i], dim=1)
#                     concat = self.conv(concat)
#         return concat


def make_merge_layer(dim):
    return MergeUp()

def make_merge2_layer(dim):
    return MergeUp2()

# def make_merge3_layer(inp_dim):
#     return MergeUp3(inp_dim)

# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    return nn.Sequential()

def make_unpool_layer(dim):
    return nn.UpsamplingBilinear2d(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

class Multiscale_DCN(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.multiscale_Dconv_layers = nn.ModuleList([
             DeformableConv2d(ic, oc, kernel_size=k_size, stride=1,
                    padding=k_size//2) for k_size in range(3,8,2)
        ])
        self.conv_1x1 = nn.Conv2d(oc*3, oc, kernel_size=1, stride=1)
    
    def forward(self, x):
        con_feature = []

        for dconv in self.multiscale_Dconv_layers:
            con_feature.append(dconv(x))

        depth_con_feature = torch.cat(con_feature, dim=1)
        
        return self.conv_1x1(depth_con_feature)
        

        

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, featuremap, H=None, W=None):
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(1, 2)
        featuremap = featuremap.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        
        q = self.q(x)
        k = self.k(featuremap)
        v = self.v(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, W, H, -1).permute(0, 3, 1, 2).contiguous()
        
        return x
    
class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_merge2_layer=make_merge2_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)
        self.merge2 = make_merge2_layer(curr_dim) # 단순 합

        if self.n > 1:
            self.attn1 = Attention(curr_dim)
            self.attn2 = Attention(curr_dim)
        else:
            self.attn1 = Attention(curr_dim)
            self.attn2 = Attention(curr_dim)
            self.attn3 = Attention(next_dim)
            self.attn4 = Attention(next_dim)
              
    def forward(self, x, feature_maps=None):
        if feature_maps is None:
            up1  = self.up1(x)
            max1 = self.max1(x)
            low1 = self.low1(max1)  # 1 : 64x64x256, 2 : 32x32x384, 3 : 16x16x384, 4 : 8x8x384, 5 : 4x4x512
            low2 = self.low2(low1)
            if len(low2) == 2: 
                low3 = self.low3(low2[0])
            else:
                low3 = self.low3(low2)          
            up2  = self.up2(low3)

            if len(low2) == 2:
                low2[1].append(up1) # 주석 예정
                low2[1].append(self.merge2(up1, up2))
                return self.merge2(up1, up2), low2[1]
            
            return self.merge2(up1, up2),  [up1, self.merge2(up1, up2)] # [up1, up2]
        else:
            for num, i in enumerate(feature_maps):  
                if x.shape[-1] == i.shape[-1] and num % 2 == 0 and x.shape[-1] != 128:
                    x = self.attn1(x, i)
                
            up1  = self.up1(x)
            max1 = self.max1(x)
            low1 = self.low1(max1)  # 1 : 64x64x256, 2 : 32x32x384, 3 : 16x16x384, 4 : 8x8x384, 5 : 4x4x512
            if low1.shape[-1] == 8:
                low1 = self.attn3(low1, feature_maps[0])
                low2 = self.low2(low1, feature_maps)
                low2 = self.attn4(low2, feature_maps[1])  
                low3 = self.low3(low2)
            else:
                low2 = self.low2(low1, feature_maps)
                low3 = self.low3(low2)
            up2  = self.up2(low3)
            for num, i in enumerate(feature_maps):    
                if x.shape[-1] == i.shape[-1] and num % 2 == 1 and x.shape[-1] != 128:
                    up2 = self.attn2(up2, i)
                    
            return self.merge2(up1, up2)
            
class GSA(nn.Module):

    def __init__(self,patch_size):
        super(GSA,self).__init__()
        
        self.key_ = nn.Linear(1,1)
        self.value_ = nn.Linear(1,1)
        self.query_ = nn.Linear(1,1)
        
        self.proj = nn.Conv2d(1, 1, kernel_size = patch_size, stride = patch_size)
        # self.proj2 = nn.Conv2d(64,1, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):

        B,C,H,W = x.shape
        down_sample = self.proj(x)  ## B, C, H/patch_size, W/patch_size
        b,c,h,w = down_sample.shape
        down_sample = down_sample.reshape(b,c,h*w).permute(0,2,1)
        x_ = x.reshape(B,C,H*W).permute(0,2,1)
        self.query = self.query_(x_) # 16384x1
        self.key   = self.key_(down_sample) # 64x1
        self.value   = self.value_(down_sample) # 64x1

        self.qk = self.query @ self.key.permute(0,2,1)
        self.qk = self.softmax(self.qk)
        
        self.qkv = self.qk @ self.value

        return self.qkv.reshape(B,1,H,W)
    
class self_attention(nn.Module):
    def __init__(self):
        super(self_attention,self).__init__()
        self.key_ = nn.Linear(1,1)
        self.query_ = nn.Linear(1,1)
        self.value_ = nn.Linear(1,1)
        self.softmax = nn.Softmax(-1)

        self.proj1 = nn.Linear(64,1)

    def forward(self,input):
        input = input.permute(0,2,1)
        self.key =self.key_(input)
        self.query =self.query_(input).permute(0,2,1)
        self.value =self.value_(input)

        self.kq = self.key @ self.query
        self.kq = self.softmax(self.kq)

        self.kqv = self.kq @ self.value
        #self.kqv = self.proj1(self.kqv)
        self.kqv = self.kqv.permute(0,2,1)

        return self.kqv

class LSA(nn.Module):
    def __init__(self,patch_size):
        super(LSA,self).__init__()
        self.patch_size = patch_size
        self.lsa_list = []
        self.self_attention = self_attention()
        self.stack_map = None

    def forward(self,patch):
        B,C,N,H,W = patch.shape
        patch = patch.squeeze()
        patch = patch.reshape(B,N,H*W)
        self.lsa_list[:N] = [self.self_attention] * N
        
        for i in range(len(self.lsa_list)):
            # self.lsa_list[i](patch[:,i,:].unsqueeze(1))
            att_patch = self.lsa_list[i]((patch[:,i,:].unsqueeze(1)))
            
            if i == 0:
                self.stack_map = att_patch
            else:
                self.stack_map = torch.cat((self.stack_map,att_patch), 1)

        return self.stack_map.unsqueeze(1).reshape(B,N,H,W)
    
class exkp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256, 
        make_tl_layer=None, make_br_layer=None,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(exkp, self).__init__()

        self.nstack    = nstack
        self.heads     = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre
        self.pre_1 = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=1)
        )
        # self.pre_2 = nn.Sequential(
        #     residual(3, 128, 256, stride=1)
        # )
        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        
        self.upfeature1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=256, out_dim=256, stride=1)
        )
        
        self.uphm = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=1, out_dim=1, stride=1, with_bn=False),
            nn.Conv2d(1, 1, (1, 1)),
        )
        self.uptwinhm = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=1, out_dim=1, stride=1, with_bn=False),
            nn.Conv2d(1, 1, (1, 1))
        )
        self.upwh = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=2, out_dim=2, stride=1, with_bn=False),
            nn.Conv2d(2, 2, (1, 1))
        )
        self.upoff = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=2, out_dim=2, stride=1, with_bn=False),
            nn.Conv2d(2, 2, (1, 1))
        )
    
        self.dcn_layers   = nn.ModuleList([
            nn.Sequential(
                Multiscale_DCN(ic, oc),
                nn.BatchNorm2d(oc),
                nn.ReLU()
            ) for ic, oc in zip([384, 384, 384, 384, 384, 384, 256, 256], 
                                [512, 512, 384, 384, 384, 384, 384, 384])
        ])

        self.conv = convolution(k=3, inp_dim=518, out_dim=256, stride=1)

        self.lsa = LSA(16)
        self.gsa = GSA(16)
        self.proj3 = nn.Conv2d(581, 256, 3, padding=1, stride = 1)
        
        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        inter_256x256 = self.pre_1(image)
        outs  = []

        for ind in range(self.nstack):
            if ind == 0:
                kp_, cnv_  = self.kps[ind], self.cnvs[ind]
                kp, feature_maps = kp_(inter) # hourglass 모듈
                cnv = cnv_(kp) # hourglass 모듈
            elif ind == 1:
                kp_, cnv_  = self.kps[ind], self.cnvs[ind]
                kp = kp_(inter_1, feature_maps) # hourglass 모듈
                cnv = cnv_(kp) # hourglass 모듈

            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                # print(layer)
                y = layer(cnv) # 히트맵, wh, offset에 대한 Loss를 구하기 위해 컨볼루션 연산 수행
                out[head] = y 

            outs.append(out)
            if ind < self.nstack - 1:
                #inter = self.inters_[ind](inter) # 중간 아래
                cnv_1 = self.cnvs_[ind](cnv) # 중간 위
                cnv_1 = self.upfeature1(cnv_1) # 업샘플링
                # cnv_1 = self.upsample(cnv_1) ff 
                # cnv_1 = self.upconv(cnv_1)
                # inter = inter_1 + cnv_1 # 스킵 커넥션
                # inter = self.relu(inter) # RELU
                
            # 패치 분할
                B, C, H, W = out['hm'].size()
                patch_size = 16
                patches = out['hm'].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
                patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
                
                lsa_out = self.lsa(patches)
                lsa_out = lsa_out.unsqueeze(1)
                input_gsa = lsa_out.view(B, C, H//patch_size, W//patch_size, patch_size, patch_size)
                input_gsa = input_gsa.permute(0, 1, 2, 4, 3, 5).contiguous()
                input_gsa = input_gsa.view(B, C, H, W)

                output_gsa = self.gsa(input_gsa)

                upgm = self.uptwinhm(output_gsa) # upGSA feature map
                uphm = self.uphm(outs[0]['hm'])
                upwh = self.upwh(outs[0]['wh'])
                upoff = self.upoff(outs[0]['reg'])
                depthconcat = torch.cat((uphm, upwh, upoff, cnv_1, inter_256x256, upgm), dim=1)
                inter = self.conv(depthconcat)
                inter_1 = self.inters[ind](inter) # 잔차블록 포워딩
                
                for i in range(8):
                    dcn = self.dcn_layers[i]
                    feature_maps[i] = dcn(feature_maps[i])
                
                
        return outs


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=2):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

def get_large_hourglass_net_multi_dcn_with_multiscale_dcn(num_layers, heads, head_conv, num_stack = 2):
    if num_stack:
        pass
    else:
        num_stack = 2
        
    model = HourglassNet(heads, num_stacks = num_stack)
    return model
