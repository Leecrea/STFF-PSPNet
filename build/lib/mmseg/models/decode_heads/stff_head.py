from .psp_head import PSPHead
from mmcv.cnn import ConvModule
import torch.nn as nn
import torch
from mmseg.registry import MODELS

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.in_planes = in_planes
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_planes, ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x):
#         print("Input to CBAM shape:", x.shape)
#         x = x * self.channel_attention(x)
#         print("After channel attention shape:", x.shape)
#         x = x * self.spatial_attention(x)
#         print("After spatial attention shape:", x.shape)
#         return x
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Global Average Pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, channels)
        
        # MLP
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        
        # Channel attention
        out = avg_out + max_out
        return torch.sigmoid(out).view(batch_size, channels, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate to form the spatial feature map
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Channel Attention
        channel_out = self.channel_attention(x)
        x = x * channel_out
        
        # Spatial Attention
        spatial_out = self.spatial_attention(x)
        x = x * spatial_out
        
        return x


@MODELS.register_module()
class CustomPSPHead(PSPHead):
    def __init__(self, **kwargs):
        super(CustomPSPHead, self).__init__(**kwargs)
        self.cbam1 = CBAM(1024)  # stage4 的特征图通道数
        self.cbam2 = CBAM(384)  # stage3 的特征图通道数
        self.cbam3 = CBAM(192)  # stage2 的特征图通道数
        self.cbam4 = CBAM(96)   # stage1 的特征图通道数

        
        print("CBAM1 input channels:", self.cbam1.channel_attention.in_channels)
        print("CBAM2 input channels:", self.cbam2.channel_attention.in_channels)
        print("CBAM3 input channels:", self.cbam3.channel_attention.in_channels)
        print("CBAM4 input channels:", self.cbam4.channel_attention.in_channels)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = ConvModule(1024 + 384, 512, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2 = ConvModule(512 + 192, 512, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv3 = ConvModule(512 + 96, 512, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # 调用父类方法获取输入特征图
        x4 = x[-1]  # stage4 的特征图
        x3 = x[-2]  # stage3 的特征图
        x2 = x[-3]  # stage2 的特征图
        x1 = x[-4]  # stage1 的特征图

        print("x4 shape before CBAM1:", x4.shape)
        x4 = self.cbam1(x4)
        print("x4 shape after CBAM1:", x4.shape)
        x4 = self.up1(x4)
    
        print("x3 shape before CBAM2:", x3.shape)
        x3 = self.cbam2(x3)
        print("x3 shape after CBAM2:", x3.shape)
        x3 = torch.cat([x4, x3], dim=1)
        x3 = self.conv1(x3)
    
        x3 = self.up2(x3)
    
        print("x2 shape before CBAM3:", x2.shape)
        x2 = self.cbam3(x2)
        print("x2 shape after CBAM3:", x2.shape)
        x2 = torch.cat([x3, x2], dim=1)
        x2 = self.conv2(x2)
    
        x2 = self.up3(x2)
    
        print("x1 shape before CBAM4:", x1.shape)
        x1 = self.cbam4(x1)
        print("x1 shape after CBAM4:", x1.shape)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv3(x1)
    
        # 将处理后的特征图传递给 PSPHead
        return super(CustomPSPHead, self).forward([x1])