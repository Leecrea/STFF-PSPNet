from .psp_head import PSPHead
from mmcv.cnn import ConvModule
import torch.nn as nn
import torch
from mmseg.registry import MODELS

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print("Input to SpatialAttention shape:", x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        print("avg_out shape:", avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        print("max_out shape:", max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        print("After cat shape:", x.shape)
        x = self.conv1(x)
        print("After conv1 shape:", x.shape)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        print("Input to CBAM shape:", x.shape)
        x = x * self.channel_attention(x)
        print("After channel attention shape:", x.shape)
        x = x * self.spatial_attention(x)
        print("After spatial attention shape:", x.shape)
        return x

@MODELS.register_module()
class CustomPSPHead(PSPHead):
    def __init__(self, **kwargs):
        super(CustomPSPHead, self).__init__(**kwargs)
        self.cbam1 = CBAM(768)
        self.cbam2 = CBAM(384)
        self.cbam3 = CBAM(192)
        self.cbam4 = CBAM(96)

        
        print("CBAM1 input channels:", self.cbam1.channel_attention.in_planes)
        print("CBAM2 input channels:", self.cbam2.channel_attention.in_planes)
        print("CBAM3 input channels:", self.cbam3.channel_attention.in_planes)
        print("CBAM4 input channels:", self.cbam4.channel_attention.in_planes)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = ConvModule(768 + 384, 512, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2 = ConvModule(512 + 192, 512, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv3 = ConvModule(512 + 96, 512, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, inputs):

        x4 = inputs[-1]
        x3 = inputs[-2]
        x2 = inputs[-3]
        x1 = inputs[-4]

        x4 = self.cbam1(x4)
        x4 = self.up1(x4)
 
        x3 = self.cbam2(x3)
        x3 = torch.cat([x4, x3], dim=1)
        x3 = self.conv1(x3)
    
        x3 = self.up2(x3)
    
        x2 = self.cbam3(x2)
        x2 = torch.cat([x3, x2], dim=1)
        x2 = self.conv2(x2)
    
        x2 = self.up3(x2)

        x1 = self.cbam4(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv3(x1)
        return super(CustomPSPHead, self).forward([x1])