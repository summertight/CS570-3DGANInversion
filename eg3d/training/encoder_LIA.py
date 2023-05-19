
import math
import torch
import torch.nn.functional as F

from torch_utils import persistence

from training.networks_stylegan2 import FullyConnectedLayer
from training.networks_stylegan2 import Conv2dLayer


@persistence.persistent_class
class Map2NoiseBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.convs = torch.nn.Sequential(
            Conv2dLayer(in_channel, in_channel, 3, bias=True, activation='lrelu'),
            Conv2dLayer(in_channel, in_channel, 3, bias=True, activation='lrelu'),
        )

        self.out_1 = torch.nn.Sequential(
            Conv2dLayer(in_channel, out_channel, 3, bias=True, activation='linear'),            
            torch.nn.InstanceNorm2d(out_channel, affine=False, track_running_stats=False),
        )

        self.out_2 = torch.nn.Sequential(
            Conv2dLayer(in_channel, out_channel, 3, bias=True, activation='linear'),            
            torch.nn.InstanceNorm2d(out_channel, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        x = self.convs(x)
        noise_1 = self.out_1(x)
        noise_2 = self.out_2(x)
        return [noise_1, noise_2]

@persistence.persistent_class
class ResBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = Conv2dLayer(in_channel, in_channel, 3, bias=True, activation='lrelu')
        self.conv2 = Conv2dLayer(in_channel, out_channel, 3, bias=True, activation='lrelu', down=2)

        self.skip = Conv2dLayer(in_channel, out_channel, 1, bias=False, activation='linear', down=2)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out
    

#----------------------------------------------------------------------------

@persistence.persistent_class
class EncoderApp(torch.nn.Module):
    def __init__(self, size, w_dim=512):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512,
            128: 256,
            256: 128,
            512: 64,
        }

        self.w_dim = w_dim
        log_size = int(math.log(size, 2))

        self.convs = torch.nn.ModuleList()
        self.convs.append(Conv2dLayer(3, channels[size], 1, bias=True, activation='lrelu'))

        #self.map2noise=torch.nn.ModuleList()
        #self.map2noise.append(Map2NoiseBlock(128, 128))
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel))
            #self.map2noise.append(Map2NoiseBlock(in_channel = out_channel, out_channel = out_channel))
            in_channel = out_channel

        self.convs.append(Conv2dLayer(in_channel, self.w_dim, 4, padding=0, bias=False, activation='linear'))
        #self.convs.append(Conv2dLayer(in_channel, self.w_dim, 4, padding=0, bias=False, activation='linear'))

    def forward(self, x):

        res = []
        h = x
        #breakpoint()
        for conv in self.convs:
        
            h = conv(h)
            
            #interm = m2m(h)
            res.append(h)
        #breakpoint()
        return res[-1].squeeze(-1).squeeze(-1), res[::-1][1:]

#----------------------------------------------------------------------------

@persistence.persistent_class
class Encoder(torch.nn.Module):
    def __init__(self, img_resolution=256, dim=512, dim_motion=20):
        super().__init__()

        # appearance netmork
        self.net_app = EncoderApp(img_resolution, dim)

        # motion network
       
        fc = [FullyConnectedLayer(512, 512*14)]
        self.fc = torch.nn.Sequential(*fc)

 

    def forward(self, input_source, input_target,):
        h_source, _ = self.net_app(input_source)
        
        h_source_14 = self.fc(h_source).reshape(-1,14,512)
        
        _, feats = self.net_app(input_target)
        
        return h_source_14, feats