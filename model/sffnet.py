import torch
import math
import torch.nn.init as init
import torch.nn as nn
import torch.fft as fft
from uncertainty_head import UncertaintyHead
from basic_module import *


class SpatialFlow(nn.Module):
    
    def __init__(self, window_size, feature_num, mid_channel, kernel_size):
        super(SpatialFlow, self).__init__()
        padding = int(kernel_size//2)
        self.Spa_CNN = nn.Sequential(nn.Conv1d(feature_num, mid_channel, kernel_size, 1, padding), nn.ReLU(),
                                     nn.Conv1d(mid_channel, feature_num, kernel_size, 1, padding))
        
    def forward(self, x):
        Spa_feature = self.Spa_CNN(x)
        return Spa_feature
    
    
class FrequencyFlow(nn.Module):
    
    def __init__(self, window_size, feature_num, mid_channel, kernel_size):
        super(FrequencyFlow, self).__init__()
        padding = int(kernel_size//2)
        self.pha_process = nn.Sequential(nn.Conv1d(feature_num, mid_channel, kernel_size, 1, padding), nn.ReLU(),
                                        nn.Conv1d(mid_channel, feature_num, kernel_size, 1, padding))
        self.amp_process = nn.Sequential(nn.Conv1d(feature_num, mid_channel, kernel_size, 1, padding), nn.ReLU(),
                                        nn.Conv1d(mid_channel, feature_num, kernel_size, 1, padding))
        
    def make_complex(self, phase, amplitude):
        real = amplitude * torch.cos(phase)
        im = amplitude * torch.sin(phase)
        complex_num = torch.complex(real, im)
        return complex_num

    def forward(self, x):
        frequency = fft.fft(x, dim=2, norm='backward')
        phase = torch.angle(frequency)
        amplitude = torch.abs(frequency)
        refine_phase = self.pha_process(phase)
        refine_amplitude = self.amp_process(amplitude)
        refine_spatial = self.make_complex(refine_phase, refine_amplitude)
        Fre_feature = torch.abs(fft.ifft(refine_spatial, dim=2, norm='backward'))
        return Fre_feature
        

class FusionBlock(nn.Module):
    
    def __init__(self, window_size, kernel_size, feature_num, r):
        super(FusionBlock, self).__init__()
        self.SA = SpatialAttention(kernel_size)
        self.CA = ChannelAttention(feature_num, r)

        
    def forward(self, fre_feature, spa_feature):
        spatial_refine_feature = self.SA(fre_feature - spa_feature)
        channel_refine_feature = self.CA(spa_feature + spatial_refine_feature)
        return channel_refine_feature
    
    
class SFFBlock(nn.Module):
    
    def __init__(self, window_size, feature_num, mid_channel, spa_ks, fre_ks, fus_ks, r, fb, sf, ff):
        super(SFFBlock, self).__init__()
        self.fb, self.sf, self.ff = fb, sf, ff
        if fb is True:
            self.FB = FusionBlock(window_size, fus_ks, feature_num, r)
        if sf is True:
            self.SF = SpatialFlow(window_size, feature_num, mid_channel, spa_ks)
        if ff is True:
            self.FF = FrequencyFlow(window_size, feature_num, mid_channel, fre_ks)
        #self.Norm = nn.BatchNorm1d(feature_num)
        #self.Norm = nn.LayerNorm(window_size)
        
    def forward(self, x):
        if self.sf is True:
            Spa_feature = self.SF(x)
        if self.ff is True:
            Fre_feature = self.FF(x)
        if self.fb is True:
            feature = self.FB(Fre_feature, Spa_feature)
        else:
            if self.sf is True:
                feature = Spa_feature
                if self.ff is True:
                    feature += Fre_feature
            else:
                if self.ff is True:
                    feature = Fre_feature
        return feature + x
    

class USFFNet(nn.Module):
    
    def __init__(self, num_block, feature_num, window_size, mid_channel_list, spa_ks_list, fre_ks_list, fus_ks_list):
        super(USFFNet, self).__init__()
        self.SFFBlock = nn.Sequential()
        for i in range(num_block):
             self.SFFBlock.add_module('SFFBlock'+str(i), SFFBlock(window_size, feature_num, mid_channel_list[i], spa_ks_list[i], fre_ks_list[i], fus_ks_list[i], 2, True, True, True))
        self.CNNI = nn.Sequential(nn.Conv1d(feature_num, 1, 3, 1, 1))
        self.Uncertainty_Head = UncertaintyHead(window_size)
    
    def forward(self, x):
        feature = self.SFFBlock(x)
        feature = self.CNNI(feature)
        feature = feature.reshape(feature.shape[0], feature.shape[1]*feature.shape[2])
        gamma, nu, alpha, beta = self.Uncertainty_Head.forward(feature)
        return gamma, nu, alpha, beta
    

class SFFNet(nn.Module):
    
    def __init__(self, num_block, feature_num, window_size, mid_channel_list, spa_ks_list, fre_ks_list, fus_ks_list):
        super(SFFNet, self).__init__()
        self.SFFBlock = nn.Sequential()
        for i in range(num_block):
             self.SFFBlock.add_module('SFFBlock'+str(i), SFFBlock(window_size, feature_num, mid_channel_list[i], spa_ks_list[i], fre_ks_list[i], fus_ks_list[i], 2, True, True, True))
        self.CNNI = nn.Sequential(nn.Conv1d(feature_num, 1, 3, 1, 1))
        self.MLP = nn.Linear(window_size, 1)
    
    def forward(self, x):
        feature = self.SFFBlock(x)
        feature = self.CNNI(feature)
        feature = feature.reshape(feature.shape[0], feature.shape[1]*feature.shape[2])
        soc = self.MLP(feature)
        return soc
    
if __name__ == '__main__':
    block_num = 5
    feature_num = 3
    window_size = 150
    mid_channel_list = [32, 16, 8, 8, 4]
    spa_ks_list = [3, 3, 3, 3, 3]
    fre_ks_list = [3, 5, 7, 7, 7]
    fus_ks_list = [3, 3, 3, 3, 3]
    model = SFFNet(block_num, feature_num, window_size, mid_channel_list, spa_ks_list, fre_ks_list, fus_ks_list)
    x = torch.ones(32, 3, 150)
    # gamma, nu, alpha, beta = model.forward(x)
    y = model.forward(x)
    print(y.shape)
