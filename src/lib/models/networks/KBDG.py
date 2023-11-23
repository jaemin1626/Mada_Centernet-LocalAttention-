import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torchvision import models
from torchvision.models.vgg import model_urls
from torch.nn import functional as F

class IndivBlur8(nn.Module):
    ## learn a kernel for each dot
    ## smaller image size and interpolation
    def __init__(self, downsample=8, s=5, softmax=False, small=False):
        super(IndivBlur8, self).__init__()
        self.downsample = downsample
        self.s = s
        self.softmax = softmax
        h = [32, 64, 128, 128]
        if small:
            h = [8, 16, 32, 32]
        
        self.adapt = nn.Sequential(
                                   nn.Conv2d(3, h[0], 3, 1, 1),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool2d(2), 
                                   nn.Conv2d(h[0], h[1], 3, 1, 1),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(h[1], h[2], 3, 1, 1),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(h[2], h[3], 3, 1, 1),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(h[3], self.s**2, 3, 1, 1))
        self._initialize_weights()

    def forward(self, points, img, shape):
        # generate kernels
        if img.shape[1] == 1:
            img = img.repeat(1,3,1,1)
        kernels = self.adapt(img)
        if self.softmax:
            kernels = F.softmax(kernels,1)
        else:
            kernels = kernels-torch.min(kernels,1,True)[0] #+ 1e-4
            kernels = kernels/torch.sum(kernels,1,True)# + 1e-12

        density = torch.zeros((shape)).cuda()
        # generate density for each image
        for j, idx in enumerate(points):
            n = len(idx) 
            if n == 0:
               continue 
            
            for i in range(n):
                y = max(0, int(idx[i,1]/self.downsample - (self.s+1)/2))
                x = max(0, int(idx[i,0]/self.downsample - (self.s+1)/2))
                ymax = min(y+self.s, density.shape[2])
                xmax = min(x+self.s, density.shape[3])
                # conv and sum
                k = kernels[0,:,min(kernels.shape[2]-1,int(idx[i,1]/16)),min(kernels.shape[3]-1,int(idx[i,0]/16))].view(1,1,self.s,self.s)
                if ymax-y < self.s or xmax-x < self.s:
                    xk, yk, xkmax, ykmax = 0, 0, self.s, self.s
                    if y == 0:
                        yk = self.s - (ymax-y)
                        ykmax = self.s
                    if x == 0:
                        xk = self.s - (xmax-x)
                        xkmax = self.s
                    if ymax == density.shape[2]:
                        ykmax = ymax - y
                        yk = 0
                    if xmax == density.shape[3]:
                        xkmax = xmax - x
                        xk = 0
                    k = k[:,:,yk:ykmax,xk:+xkmax]
                density[j,:,y:ymax,x:xmax] += k[0]

        return density

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

__all__ = ['vgg19']
model_urls = {
    'vgg16': 'http://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'http://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'http://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'http://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, down=8):
        super(VGG, self).__init__()
        self.down = down
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        if self.down < 16:
            x = F.interpolate(x, scale_factor=2)
        x = self.reg_layer(x)
        x = torch.abs(x)
        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    # in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],    
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],    
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], batch_norm=False))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model


def get_kbdg(num_layers, heads, head_conv, num_stack = 2):
    if num_stack:
        pass
    else:
        num_stack = 2
        
    refiner = IndivBlur8()
    model = vgg19()
    return model, refiner