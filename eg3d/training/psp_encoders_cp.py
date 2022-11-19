import numpy as np
import torch, math
import torch.nn.functional as F
from torch import nn
#from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module, MaxPool2d, AdaptiveAvgPool2d, AdaptiveAvgPool2d, Sigmoid, ReLU
import sys
from collections import namedtuple
#
#sys.path.append('..')
#from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
#from training.op import fused_leaky_relu
##XXX from models.stylegan2.model import EqualLinear

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
	""" A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
	return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
	if num_layers == 50:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 100:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=13),
			get_block(in_channel=128, depth=256, num_units=30),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 152:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=8),
			get_block(in_channel=128, depth=256, num_units=36),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	else:
		raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
	return blocks

class SEModule(nn.Module):
	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x

class bottleneck_IR_SE(nn.Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR_SE, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = nn.MaxPool2d(1, stride)
		else:
			self.shortcut_layer = nn.Sequential(
				nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				nn.BatchNorm2d(depth)
			)
		self.res_layer = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
			nn.PReLU(depth),
			nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
			nn.BatchNorm2d(depth),
			SEModule(depth, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut

class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        #if self.activation:
        #    out = F.linear(input, self.weight * self.scale)
        #    out = fused_leaky_relu(out, self.bias * self.lr_mul)

        #else:
        out = F.linear(
        input, self.weight * self.scale, bias=self.bias * self.lr_mul
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )



class GradualNoiseBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GradualNoiseBlock, self).__init__()
        self.convs = torch.nn.Sequential(
            #Conv2dLayer(in_channel, in_channel, 3, bias=True, activation='lrelu'),
            #Conv2dLayer(in_channel, in_channel, 3, bias=True, activation='lrelu'),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            
        )

        self.out_1 = torch.nn.Sequential(
            #Conv2dLayer(in_channel, out_channel, 3, bias=True, activation='linear'),    
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),        
            nn.InstanceNorm2d(out_channel, affine=False, track_running_stats=False),
        )

        self.out_2 = torch.nn.Sequential(
            #Conv2dLayer(in_channel, out_channel, 3, bias=True, activation='linear'),   
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),         
            nn.InstanceNorm2d(out_channel, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        x = self.convs(x)
        noise_1 = self.out_1(x)
        noise_2 = self.out_2(x)
        return [noise_1, noise_2]

class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(nn.Module):
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
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),  # opts.input_nc = 3
                                      nn.BatchNorm2d(64),
                                      nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)
        if w_type == 'w+':
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
     
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, src, tgt=None):
        if tgt is None:
            x = src
        else:
            #breakpoint()
            x = torch.cat((src, tgt),axis=0)
            #pass
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

            return out, out_maps

    
                #pass
        return out
    def encode(self, x_src, x_tgt):
        return self.forward(x_src, x_tgt)
        

