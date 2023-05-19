import torch.nn as nn
import torch
class ExpMappingNet(nn.Module):
    def __init__(self, coeff_nc=32, descriptor_nc=512, num_layers=3, residual_layers=[1]):
        super(ExpMappingNet, self).__init__()

        self.num_layers = num_layers
        self.residual_layers = residual_layers
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            nn.Linear(coeff_nc, descriptor_nc, bias=True))
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            if i in residual_layers:
                net = nn.Sequential(nonlinearity,
                nn.Linear(descriptor_nc+coeff_nc, descriptor_nc, bias=True))
                self.layers.append(net)
            else:
                net = nn.Sequential(nonlinearity,
                    nn.Linear(descriptor_nc, descriptor_nc, bias=True))
                self.layers.append(net)
        self.final = nn.Sequential(nonlinearity,
                    nn.Linear(descriptor_nc, descriptor_nc*14, bias=True))
       

        # self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, exp):
        
        out = self.first(exp)
        for i in range(self.num_layers):
            if i in self.residual_layers:
                out = self.layers[i](torch.cat([out,exp],-1))
            else:
                out = self.layers[i](out)
            
        out = self.final(out)
        return out.reshape(-1,14,512)  