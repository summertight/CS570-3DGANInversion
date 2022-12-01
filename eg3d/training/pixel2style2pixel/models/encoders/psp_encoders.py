import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
import sys
#
#sys.path.append('..')
from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear


class GradualNoiseBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GradualNoiseBlock, self).__init__()
        self.convs = torch.nn.Sequential(
            #Conv2dLayer(in_channel, in_channel, 3, bias=True, activation='lrelu'),
            #Conv2dLayer(in_channel, in_channel, 3, bias=True, activation='lrelu'),
            Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            
        )

        self.out_1 = torch.nn.Sequential(
            #Conv2dLayer(in_channel, out_channel, 3, bias=True, activation='linear'),    
            Conv2d(in_channel, out_channel, kernel_size=3, padding=1),        
            torch.nn.InstanceNorm2d(out_channel, affine=False, track_running_stats=False),
        )

        self.out_2 = torch.nn.Sequential(
            #Conv2dLayer(in_channel, out_channel, 3, bias=True, activation='linear'),   
            Conv2d(in_channel, out_channel, kernel_size=3, padding=1),         
            torch.nn.InstanceNorm2d(out_channel, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        x = self.convs(x)
        noise_1 = self.out_1(x)
        noise_2 = self.out_2(x)
        return [noise_1, noise_2]

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers=50, mode='ir_se', n_styles=14, img_res=256, block_selections=[], w_type = 'w+'):
        super(GradualStyleEncoder, self).__init__()
        assert n_styles in [14, 20]
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        assert w_type in ['w+', 'w++']
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),  # opts.input_nc = 3
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        if w_type == 'w+':
            n_styles=14
            self.styles = nn.ModuleList()
            self.style_count = n_styles  # int(math.log(self.opts.output_size, 2)) * 2 - 2     # 
            self.coarse_ind = 3
            self.middle_ind = 7
            #breakpoint()
        elif w_type == 'w++':
            
            n_styles = 20
            self.styles = nn.ModuleList()
            self.style_count = n_styles  # int(math.log(self.opts.output_size, 2)) * 2 - 2     # 
            self.coarse_ind = 5
            self.middle_ind = 11
            #breakpoint()
        for i in range(self.style_count):
            if i < self.coarse_ind:# 0,1,2
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:# 3,4,5,6
                style = GradualStyleBlock(512, 512, 32)
            else:# XXX 7,8,9,10,11,12,13
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        #breakpoint()
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        #breakpoint()
        # XXX From StyleGAN2
        self.block_selections = list(map(int,block_selections))
        
        self.img_resolution_log2 = int(np.log2(img_res))
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channel_base    = 32768   #XXX 2^15 # Overall multiplier for the number of channels.
        channel_max     = 512 
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        #import pdb;pdb.set_trace()
        
        if len(self.block_selections)!= 0:
            self.noises = nn.ModuleList()
            for res in self.block_resolutions:
                in_channels = channels_dict[res // 2] if res > 4 else 0
                out_channels = channels_dict[res]
                print(in_channels, out_channels)
                if res in self.block_selections:
                    self.noises.append(GradualNoiseBlock(512, out_channels))

        

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x #XXX 512,16,16 # XXX 0,1,2 (1,16,16)으로 두번뽑기
            elif i == 20:
                c2 = x #XXX 256, 32, 32 # XXX 3,4,5,6 (1,32,32)으로 두번뽑기
            elif i == 23:
                c3 = x #XXX 128, 64, 64
        #import pdb; pdb.set_trace()
        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
        #breakpoint()
        out = torch.stack(latents, dim=1)

        if len(self.block_selections)!= 0:
            #breakpoint()
            out_maps = []
            #out_maps = []
            for i in range(1,8):

                if i==1:
                    out_maps.append(None)

                elif 2**(i+1) == self.block_selections[0]:
                    out_maps += self.noises[0](c3)

                elif 2**(i+1) == self.block_selections[1]:
                    out_maps += self.noises[1](p2)

                else:
                    out_maps.append(None); out_maps.append(None)
            #breakpoint()
            #for idx, k in enumerate(self.block_selections):
                #breakpoint()
            #    if idx==0:
            #        out_maps.append(self.noises[idx](c3))
            #    if idx==1:
            #        out_maps.append(self.noises[idx](p2))
            
            return out, out_maps

    
                #pass
        return out
    def encode(self,x):
        return self.forward(x)
        

class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x
