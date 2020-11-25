"""vnet architecture"""
import torch
from torch import nn
from torch.nn import Module, Conv3d, Parameter
import numpy 

class GroupNorm3D(Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm3D, self).__init__()
        self.weight = Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W, D = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W, D)
        
        return x * self.weight + self.bias 


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization=None, activation=None, expand_chan=False, drop_out_need = True):
        super(ResidualConvBlock, self).__init__()

        self.expand_chan = expand_chan
        self.drop_out_need = drop_out_need
        if self.expand_chan:                                                    ## not yet
            pass
        ops = []
        for i in range(n_stages):
            if normalization:
                ops.append(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1))
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
                if normalization == 'groupnorm':
                    ops.append(GroupNorm3D(n_filters_out))
                if normalization == 'instancenorm':
                    ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                ops.append(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1))
            if activation:
                if activation == 'ReLU':
                    ops.append(nn.ReLU(inplace=True))
                if activation == 'LeakyReLU':
                    ops.append(nn.LeakyReLU(inplace=True))
                if activation == 'PReLU':
                    ops.append(nn.PReLU(inplace=True))
            else:
                ops.append(nn.ReLU(inplace=True))
            
        self.conv = nn.Sequential(*ops)
        self.drop_out = nn.Dropout3d(0.2)                                        ## dropout

    def forward(self, x):
        
        if self.expand_chan:                                                     ## not yet
            x = self.conv(x)
        else:
            x = (self.conv(x) + x)    
        if self.drop_out_need:
            x = self.drop_out(x)
            
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization=None, activation=None):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
            if normalization == 'groupnorm':
                    ops.append(GroupNorm3D(n_filters_out))
            if normalization == 'instancenorm':
                    ops.append(nn.InstanceNorm3d(n_filters_out))
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if activation:
            if activation == 'ReLU':
                 ops.append(nn.ReLU(inplace=True))
            if activation == 'LeakyReLU':
                ops.append(nn.LeakyReLU(inplace=True))
            if activation == 'PReLU':
                ops.append(nn.PReLU(inplace=True))
        else:
            ops.append(nn.ReLU(inplace=True))


        self.conv = nn.Sequential(*ops)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization=None, activation=None):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
            if normalization == 'groupnorm':
                    ops.append(GroupNorm3D(n_filters_out))
            if normalization == 'instancenorm':
                    ops.append(nn.InstanceNorm3d(n_filters_out))
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if activation:
            if activation == 'ReLU':
                 ops.append(nn.ReLU(inplace=True))
            if activation == 'LeakyReLU':
                ops.append(nn.LeakyReLU(inplace=True))
            if activation == 'PReLU':
                ops.append(nn.PReLU(inplace=True))
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        self.conv = nn.Sequential(*ops)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class VNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=16, normalization=None, activation=None, depth = 3):
        super(VNet, self).__init__()
        self.depth = depth
        if n_channels > 1:
            self.block1 = ResidualConvBlock(1, n_channels, n_filters, normalization=normalization, activation=activation, expand_chan=True)
        else:
            self.block1 = ResidualConvBlock(1, n_channels, n_filters, normalization=normalization, activation=activation)
        
        self.block1_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization, activation=activation)
        
        self.block2 = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block2_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization, activation=activation)

        self.block3 = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block3_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization, activation=activation)
        self.block3_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, activation=activation)

        self.block4 = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization, activation=activation, drop_out_need = False)
        self.block4_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization, activation=activation)
        
        self.block5 = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization, activation=activation)
        self.block5_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization, activation=activation)
        
        
        self.block6 = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization, activation=activation)
        self.block6_up = UpsamplingDeconvBlock(n_filters * 2, n_filters , normalization=normalization, activation=activation)
        
        self.block7 = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization, activation=activation)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        
            
    def forward(self, input):
        x1 = self.block1(input)
        x1_dw = self.block1_dw(x1)
        
        x2 = self.block2(x1_dw)
        x2_dw = self.block2_dw(x2)

        x3 = self.block3(x2_dw)
        if self.depth == 3:
            x3_dw = self.block3_dw(x3)
            x4 = self.block4(x3_dw)
            x4_up = self.block4_up(x4)
            x4_up = x4_up + x3
            x5 = self.block5(x4_up)
            x5_up = self.block5_up(x5)
            x5_up = x5_up + x2
            x3_up = x5_up
        else:
            x3_up = self.block3_up(x3)
            x3_up = x3_up + x2
        
        x6 = self.block6(x3_up)
        x6_up = self.block6_up(x6)
        x6_up = x6_up + x1

        x7 = self.block7(x6_up)
        
        out = self.out_conv(x7)
        
        return out
   
    def enable_test_dropout(self): 
        attr_dict = self.__dict__['_modules']
        for i in range(1, self.depth+1):
            encode_block, decode_block = attr_dict['block' + str(i)], attr_dict['block' + str(8-i)]
            encode_block.drop_out = encode_block.drop_out.apply(nn.Module.train)
            decode_block.drop_out = decode_block.drop_out.apply(nn.Module.train) 
            
            
    def predict(self, X, device=None, enable_dropout=False):  
        self.eval()

        if type(X) is numpy.ndarray:
            X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device)    
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device)
            
        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(X)

        return out
