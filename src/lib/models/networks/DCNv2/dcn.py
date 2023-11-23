import torch
import torchvision.ops
from torch import nn

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x
    


class Perpendicular_DeformableConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(Perpendicular_DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.frames = nn.Parameter(torch.randn((out_dim, kernel_size * kernel_size * 2)),True)
        self.weights = nn.ModuleList([nn.Conv2d(in_dim,1,kernel_size,stride=kernel_size) for _ in range(out_dim)])
        self.bn   = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        _,_,height,width = x.shape
        _,offset_len = self.frames.shape
        x_len = (offset_len//2)

        y_idx,x_idx = torch.meshgrid(torch.arange(height,requires_grad=False,device=torch.device('cuda')), torch.arange(width,requires_grad=False,device=torch.device('cuda')))
        
        out = []

        for idx,frame in enumerate(self.frames):
            frame = frame.repeat(height,width,1)
            frame[:,:,:x_len] = frame[:,:,:x_len]+x_idx.unsqueeze(-1)
            frame[:,:,x_len:] = frame[:,:,x_len:]+y_idx.unsqueeze(-1)

            q_lt = frame.detach().floor()
            q_rb = q_lt + 1

            q_lt = torch.cat([torch.clamp(q_lt[..., :x_len], 0, width-1), torch.clamp(q_lt[..., x_len:], 0, height-1)], dim=-1).long()
            q_rb = torch.cat([torch.clamp(q_rb[..., :x_len], 0, width-1), torch.clamp(q_rb[..., x_len:], 0, height-1)], dim=-1).long()
            q_lb = torch.cat([q_lt[..., :x_len], q_rb[..., x_len:]], dim=-1)
            q_rt = torch.cat([q_rb[..., :x_len], q_lt[..., x_len:]], dim=-1)

            # clip p
            frame = torch.cat([torch.clamp(frame[..., :x_len], 0, width-1), torch.clamp(frame[..., x_len:], 0, height-1)], dim=-1)

            g_lt = (1 + (q_lt[..., :x_len].type_as(frame) - frame[..., :x_len])) * (1 + (q_lt[..., x_len:].type_as(frame) - frame[..., x_len:]))
            g_rb = (1 - (q_rb[..., :x_len].type_as(frame) - frame[..., :x_len])) * (1 - (q_rb[..., x_len:].type_as(frame) - frame[..., x_len:]))
            g_lb = (1 + (q_lb[..., :x_len].type_as(frame) - frame[..., :x_len])) * (1 - (q_lb[..., x_len:].type_as(frame) - frame[..., x_len:]))
            g_rt = (1 - (q_rt[..., :x_len].type_as(frame) - frame[..., :x_len])) * (1 + (q_rt[..., x_len:].type_as(frame) - frame[..., x_len:]))

            x_q_lt = x[:,:,q_lt[:,:,x_len:],q_lt[:,:,:x_len]]
            x_q_rb = x[:,:,q_rb[:,:,x_len:],q_rb[:,:,:x_len]]
            x_q_lb = x[:,:,q_lb[:,:,x_len:],q_lb[:,:,:x_len]]
            x_q_rt = x[:,:,q_rt[:,:,x_len:],q_rt[:,:,:x_len]]

            # (b, c, h, w, N)
            x_offset = g_lt * x_q_lt[:,:] + \
                    g_rb * x_q_rb[:,:] + \
                    g_lb * x_q_lb[:,:] + \
                    g_rt * x_q_rt[:,:]
            
            x_offset = self._reshape_x_offset(x_offset,self.kernel_size)
            
            out.append(self.weights[idx](x_offset))
        out = torch.cat(out,dim=1)
        return self.relu(self.bn(out))


    def _reshape_x_offset(self, x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset