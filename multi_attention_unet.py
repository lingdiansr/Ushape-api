import torch.nn as nn
import torch.nn.functional as F
import torch

###############################
class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.dilation_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=d, dilation=d)
            for d in [1, 2, 3, 6]
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        ms_features = [conv(x) for conv in self.dilation_convs]
        concat_feat = torch.cat(ms_features + [x], dim=1)
        fused = self.fusion(concat_feat)

        # 通道注意力
        c_att = self.channel_att(fused)
        c_out = fused * c_att

        # 空间注意力
        s_att = self.spatial_att(c_out)
        return x + c_out * s_att

######################################
class AttentionUNet(torch.nn.Module):
    def __init__(self, input_channels, class_number, **kwargs):
        super().__init__()
        down_channel = kwargs['down_channel']

        # 增加中间通道数以适应多尺度模块
        self.inc = InConv(input_channels, down_channel)
        self.attn1 = MultiScaleAttention(down_channel)  # 新增注意力

        self.down1 = DownLayer(down_channel, down_channel * 2)
        self.attn2 = MultiScaleAttention(down_channel * 2)  # 新增注意力

        self.down2 = DownLayer(down_channel * 2, down_channel * 2)
        self.attn3 = MultiScaleAttention(down_channel * 2)  # 新增注意力

        self.up1 = UpLayer(down_channel * 4, down_channel)
        self.up2 = UpLayer(down_channel * 2, down_channel)
        self.outc = OutConv(down_channel, class_number)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.attn1(x1)  # 应用注意力

        x2 = self.down1(x1)
        x2 = self.attn2(x2)  # 应用注意力

        x3 = self.down2(x2)
        x3 = self.attn3(x3)  # 应用注意力

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x).permute(0, 2, 3, 1).contiguous()
#########################################################

class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch),
                                         nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.double_conv(x)
        return x

#########################
class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.MaxPool2d(kernel_size=1),
            DoubleConv(in_ch, out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        # 修改残差路径卷积参数
        self.shortcut = nn.Sequential(
            nn.MaxPool2d(kernel_size=1),  # 保持相同下采样方式
            # nn.MaxPool2d(2),  # 保持相同下采样方式
            nn.Conv2d(in_ch, out_ch, 1)  # 仅调整通道数
        )

    def forward(self, x):
        return self.down_conv(x) + self.shortcut(x)

#########################################
class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpLayer(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpLayer, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
                        diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x