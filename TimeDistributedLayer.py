import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

class TimeDistributedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dropout=False):
        super(TimeDistributedConv2d, self).__init__()
        model = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        
        if dropout:
            model.append(nn.Dropout(0.5, inplace=True))
        
        self.conv2d = nn.Sequential(*model)

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.conv2d(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)

        y = self.conv2d(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), -1, x.size(-2), x.size(-1))  # (samples, timesteps, output_size)
        return y

# class TimeDistributedConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dropout=False):
#         super(TimeDistributedConv2d, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

#     def forward(self, x):
#         if len(x.size()) <= 2:
#             return self.conv2d(x)

#         # Squash samples and timesteps into a single axis
#         x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)

#         y = self.conv2d(x_reshape)
#         y = y.contiguous().view(x.size(0), x.size(1), -1, x.size(-2), x.size(-1))  # (samples, timesteps, output_size)
#         return y

class TimeDistributedMaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(TimeDistributedMaxPool, self).__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.maxpool2d(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)

        y = self.maxpool2d(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(-2)//2, x.size(-1)//2)  # (samples, timesteps, output_size)
        return y

# self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
class TimeDistributedUpsampling(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(TimeDistributedUpsampling, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=scale_factor, mode=mode)
        
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.upsampling(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3),x.size(-2), x.size(-1))     # (samples * timesteps, input_size)

        y = self.upsampling(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(-2)*2, x.size(-1)*2)  # (samples, timesteps, output_size)
        return y
