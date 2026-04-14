import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# ----------- Channel Attention -----------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.global_avg_pool(x)
        w = self.fc(w)
        return x * w


# ----------- Spatial Attention -----------
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


# ----------- Multi-Scale Attention Module -----------
class MSAM(nn.Module):
    def __init__(self, in_channels):
        super(MSAM, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ----------- ASPP + MSAM -----------
class ASPP_MSAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP_MSAM, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),  # 替代 BatchNorm
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.attention = MSAM(out_channels)

    def forward(self, x):
        size = x.shape[2:]
        out = [block(x) for block in self.aspp_blocks]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=True)
        out.append(gp)
        out = torch.cat(out, dim=1)
        out = self.project(out)
        out = self.attention(out)
        return out


# ----------- Feature Fusion Module -----------
class FeatureFusion(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + high_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_feat, high_feat):
        low_feat = self.low_proj(low_feat)
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=True)
        return self.fuse(torch.cat([low_feat, high_feat], dim=1))


# ----------- DeepLabV3+ with MSAM and ResNet18 -----------
class DeepLabV3Plus_MSAM(nn.Module):
    def __init__(self, n_classes=5):  # 默认输出类别为 5 类（0~4）
        super(DeepLabV3Plus_MSAM, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 64
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

        self.aspp_msam = ASPP_MSAM(in_channels=512, out_channels=256)
        self.fusion = FeatureFusion(low_channels=64, high_channels=256, out_channels=256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        size = x.shape[2:]

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x_aspp = self.aspp_msam(x4)
        x = self.fusion(x1, x_aspp)
        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)

        return x  # 输出为 (N, 5, H, W)，配合 CrossEntropyLoss(ignore_index=255)
