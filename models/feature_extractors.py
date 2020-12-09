import torch
import torch.nn as nn
from .utils import *

class NatureDQNCNN(nn.Module):
    def __init__(self, h, w, feat_dim=512, in_channel=12, out_channels=[32,64,64], kernel_sizes=[8,4,3], strides=[4,2,1], img_format='NHWC'):
        # Assumes NCHW format
        super(NatureDQNCNN, self).__init__()
        if img_format not in ['NCHW', 'NHWC']:
            raise TypeError(f'Unsupported image data format: {img_format}')
        assert len(out_channels) == len(kernel_sizes) == len(strides), \
            "len of out_channels, kernel_sizes, strides should match" 
        num_layers = len(kernel_sizes)
        channels = [in_channel] + out_channels
        convs = []
        convw, convh = w, h
        for i in range(num_layers):
            convs.append(nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i], strides[i]))
            convs.append(nn.ReLU())
            convw = conv2d_size_out(convw, kernel_sizes[i], strides[i])
            convh = conv2d_size_out(convh, kernel_sizes[i], strides[i])

        self.convs = nn.Sequential(*convs)
        latent_size = convw*convh*channels[-1]

        self.fcs = nn.Sequential(
            nn.Linear(latent_size, feat_dim),
            nn.ReLU()
        )
        self.feat_dim = feat_dim

        self._need_permute = img_format != 'NCHW'

    def forward(self, x):
        if self._need_permute:
            x = x.permute(0, 3, 1, 2)
        x = x / 255.0        
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

class ImpalaCNN(nn.Module):
    def __init__(self,
                 h,
                 w,
                 feat_dim=256,
                 in_channel=12,
                 dropout=0.0,
                 batch_norm=False,
                 impala_size='small',
                 img_format='NHWC'):
        super(ImpalaCNN, self).__init__()
        # Assumes NCHW format
        if img_format not in ['NCHW', 'NHWC']:
            raise TypeError(f'Unsupported image data format: {img_format}')        
        if impala_size not in ['vanilla', 'small', 'large']:
            raise ValueError(f'impala_size should be '
                             f'one of \'vanilla\' or \'small\' or \'large\'! '
                             f'Got {impala_size} instead!')
        if impala_size == 'vanilla':
            channel_groups = [4, 4]
        elif impala_size == 'small':
            channel_groups = [16, 32, 32]
        else:
            channel_groups = [32, 64, 64, 64, 64]
        
        convw, convh = w, h
        num_layers = len(channel_groups)
        for _ in range(num_layers):
            convw = conv2d_size_out(convw, 3, 2, 1)
            convh = conv2d_size_out(convh, 3, 2, 1)
        
        cnn_out_size = convw*convh*channel_groups[-1]
        
        self.convs = nn.ModuleList()    
        for ch in channel_groups:
            self.convs.append(
                ImpalaConvBlock(in_channels=in_channel,
                                out_channels=ch,
                                dropout=dropout,
                                batch_norm=batch_norm)
            )
            self.convs.append(
                nn.MaxPool2d(kernel_size=3,
                             stride=2,
                             padding=1)
            )
            self.convs.append(
                ImpalaResidualBlock(num_channels=ch,
                                    dropout=dropout,
                                    batch_norm=batch_norm)
            )
            self.convs.append(
                ImpalaResidualBlock(num_channels=ch,
                                    dropout=dropout,
                                    batch_norm=batch_norm)
            )
            in_channel = ch
        self.fcs = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=cnn_out_size, out_features=feat_dim),
            nn.ReLU()
        )
        self.feat_dim = feat_dim

        self._need_permute = img_format != 'NCHW'

    def forward(self, x):
        if self._need_permute:
            x = x.permute(0, 3, 1, 2)
        x = x / 255.0                   
        for layer in self.convs:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

class ImpalaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=0.0, batch_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1)
        )
        if dropout > 0.0:
            self.layers.append(
                nn.Dropout2d(p=dropout)
            )
        if batch_norm:
            self.layers.append(
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ImpalaResidualBlock(nn.Module):
    def __init__(self, num_channels,
                 dropout=0.0, batch_norm=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            ImpalaConvBlock(in_channels=num_channels,
                            out_channels=num_channels,
                            dropout=dropout,
                            batch_norm=batch_norm),
            nn.ReLU(),
            ImpalaConvBlock(in_channels=num_channels,
                            out_channels=num_channels,
                            dropout=dropout,
                            batch_norm=batch_norm)
        )

    def forward(self, x):
        out = self.layers(x)
        new_out = out + x
        return new_out
