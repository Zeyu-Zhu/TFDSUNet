import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.CNN = nn.Conv1d(2, 1, kernel_size, 1, kernel_size//2)
    
    # (batch_size, feature_num, window_size)
    def forward(self, x):
        batch_size, feature_num, window_size = x.shape
        feature_max, _ = torch.max(x, dim=1)
        feature_avg = torch.sum(x, dim=1) / feature_num
        refine_weight = self.CNN(torch.stack([feature_max, feature_avg], dim=1))
        refine_feature = refine_weight*x
        return refine_feature
    
        
class ChannelAttention(nn.Module):
    
    def __init__(self, feature_num, r):
        super(ChannelAttention, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(feature_num, feature_num*r), nn.Linear(feature_num*r, feature_num))
        self.Sigmoid = nn.Sigmoid()
        
    # (batch_size, feature_num, window_size)
    def forward(self, x):
        batch_size, feature_num, window_size = x.shape
        feature_max, _ = torch.max(x, dim=2)
        feature_avg = torch.sum(x, dim=2) / window_size
        refine_weigth = self.Sigmoid(self.MLP(feature_max)+self.MLP(feature_avg))
        refine_weigth = refine_weigth.unsqueeze(dim=-1)
        refine_feature = refine_weigth*x
        return refine_feature
        
        
if __name__ == '__main__':
    model = ChannelAttention(3, 3)
    x = torch.ones((32, 3, 100))
    out = model.forward(x)
    print(out.shape)