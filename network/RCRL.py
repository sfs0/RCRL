import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        # 增加avg_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # -------
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 双路并行
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out  # 特征融合
        return self.sigmoid(out)


# 修改后：
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=padding, bias=False
        )  # 通道数改为2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用平均池化和最大池化的组合
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接两种池化结果
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Refine2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Refine2, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class HRA_Fuse1(nn.Module):

    def __init__(self, channel):
        super(HRA_Fuse1, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class HRA_Fuse2(nn.Module):

    def __init__(self, channel):
        super(HRA_Fuse2, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

        # 新增变化注意力模块
        self.change_attn = nn.Sequential(
            nn.Conv2d(3 * channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        # 新增变化注意力
        change_attn = self.change_attn(x3_2)
        x3_2 = x3_2 * change_attn + x3_2  # 残差连接

        return x3_2


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )

    def forward(self, attention, x1, x2, x3):
        x1 = x1 + torch.mul(x1, self.upsample4(attention))
        x2 = x2 + torch.mul(x2, self.upsample2(attention))
        x3 = x3 + torch.mul(x3, attention)

        return x1, x2, x3


# --------------------------------
class FeatureInteraction(nn.Module):
    """特征交互模块，生成多种特征交互表示"""

    def __init__(self, in_planes):
        super(FeatureInteraction, self).__init__()
        self.conv_diff = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
        )
        self.conv_sum = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(2 * in_planes, in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # 计算差异特征 (突出变化区域)
        diff = torch.abs(x1 - x2)
        diff = self.conv_diff(diff)

        # 计算和特征 (保留共同特征)
        sum_feat = x1 + x2
        sum_feat = self.conv_sum(sum_feat)

        # 计算拼接特征 (保留各自特征)
        cat = torch.cat([x1, x2], dim=1)
        cat = self.conv_cat(cat)

        return diff, sum_feat, cat


# --------------------


# Ghost and ASPP modules
class GhostConv(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=2):
        super().__init__()
        init_ch = int(out_ch / ratio)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, 1, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True),
        )
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_ch, out_ch, 3, padding=1, groups=init_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.primary_conv(x)
        return self.cheap_conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=[6, 12, 18]):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        )
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.project = nn.Sequential(
            nn.Conv2d(len(self.branches) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        results = []
        for branch in self.branches:
            results.append(
                F.interpolate(branch(x), size=size, mode="bilinear", align_corners=True)
            )
        return self.project(torch.cat(results, dim=1))


# --------------------------------


class MCAP_Head(nn.Module):
    def __init__(self, in_channels, compressed_channels, output_channels=1):
        """
        变化感知输出头
        Args:
            in_channels: 输入通道数 (3 * channel)
            compressed_channels: 压缩后通道数 (2 * channel)
            output_channels: 输出通道数
        """
        super(MCAP_Head, self).__init__()

        # 第一阶段：通道压缩和特征提炼
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(in_channels, compressed_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(compressed_channels),
            nn.ReLU(inplace=True),
            GhostConv(compressed_channels, compressed_channels // 2),
            nn.BatchNorm2d(compressed_channels // 2),
            nn.ReLU(inplace=True),
        )

        # 第二阶段：多尺度上下文提取
        self.multiscale_context = ASPP(
            compressed_channels // 2, compressed_channels // 4
        )

        # 第三阶段：最终变化预测
        self.change_prediction = nn.Sequential(
            nn.Conv2d(
                compressed_channels // 4,
                compressed_channels // 8,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(compressed_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(compressed_channels // 8, output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.feature_refinement(x)
        x = self.multiscale_context(x)
        x = self.change_prediction(x)
        return x


class SemiModel(nn.Module):
    def __init__(self, channel=32):
        super(SemiModel, self).__init__()

        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.HL3 = Refine2(256, channel)
        self.HL4 = Refine2(512, channel)
        self.HL5 = Refine2(512, channel)
        self.H1 = HRA_Fuse1(channel)

        self.HL1 = Refine2(64, channel)
        self.HL2 = Refine2(128, channel)
        self.HL3_2 = Refine2(channel, channel)
        self.H2 = HRA_Fuse2(channel)
        self.HA = Refine()

        self.atten_A_channel_1 = ChannelAttention(64)
        self.atten_A_channel_2 = ChannelAttention(128)
        self.atten_A_channel_3 = ChannelAttention(256)
        self.atten_A_channel_4 = ChannelAttention(512)
        self.atten_A_channel_5 = ChannelAttention(512)

        self.atten_A_spatial_1 = SpatialAttention()
        self.atten_A_spatial_2 = SpatialAttention()
        self.atten_A_spatial_3 = SpatialAttention()
        self.atten_A_spatial_4 = SpatialAttention()
        self.atten_A_spatial_5 = SpatialAttention()

        self.atten_B_channel_1 = ChannelAttention(64)
        self.atten_B_channel_2 = ChannelAttention(128)
        self.atten_B_channel_3 = ChannelAttention(256)
        self.atten_B_channel_4 = ChannelAttention(512)
        self.atten_B_channel_5 = ChannelAttention(512)

        self.atten_B_spatial_1 = SpatialAttention()
        self.atten_B_spatial_2 = SpatialAttention()
        self.atten_B_spatial_3 = SpatialAttention()
        self.atten_B_spatial_4 = SpatialAttention()
        self.atten_B_spatial_5 = SpatialAttention()

        # 添加特征交互模块
        self.feature_interaction1 = FeatureInteraction(64)
        self.feature_interaction2 = FeatureInteraction(128)
        self.feature_interaction3 = FeatureInteraction(256)
        self.feature_interaction4 = FeatureInteraction(512)
        self.feature_interaction5 = FeatureInteraction(512)

        self.output_head = MCAP_Head(
            in_channels=3 * channel, compressed_channels=2 * channel, output_channels=1
        )
        # ----------------------------

    def forward(self, A, B):
        layer1_A = self.inc(A)
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)
        layer4_A = self.down3(layer3_A)
        layer5_A = self.down4(layer4_A)

        layer1_B = self.inc(B)
        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)
        layer4_B = self.down3(layer3_B)
        layer5_B = self.down4(layer4_B)

        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))

        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))

        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))

        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))

        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))

        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))

        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))

        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))

        layer5_A = layer5_A.mul(self.atten_A_channel_5(layer5_A))
        layer5_A = layer5_A.mul(self.atten_A_spatial_5(layer5_A))

        layer5_B = layer5_B.mul(self.atten_B_channel_5(layer5_B))
        layer5_B = layer5_B.mul(self.atten_B_spatial_5(layer5_B))

        # 使用特征交互模块
        diff1, sum1, cat1 = self.feature_interaction1(layer1_A, layer1_B)
        diff2, sum2, cat2 = self.feature_interaction2(layer2_A, layer2_B)
        diff3, sum3, cat3 = self.feature_interaction3(layer3_A, layer3_B)
        diff4, sum4, cat4 = self.feature_interaction4(layer4_A, layer4_B)
        diff5, sum5, cat5 = self.feature_interaction5(layer5_A, layer5_B)
        # 使用差异特征作为主要变化信号，但保留一些共同特征
        alpha = 0.7  # 差异特征的权重
        layer1 = alpha * diff1 + (1 - alpha) * sum1
        layer2 = alpha * diff2 + (1 - alpha) * sum2
        layer3 = alpha * diff3 + (1 - alpha) * sum3
        layer4 = alpha * diff4 + (1 - alpha) * sum4
        layer5 = alpha * diff5 + (1 - alpha) * sum5
        # ------------------------------------------------------------------------------

        layer3 = self.HL3(layer3)
        layer4 = self.HL4(layer4)
        layer5 = self.HL5(layer5)
        attention_map = self.H1(layer5, layer4, layer3)

        layer1, layer2, layer3 = self.HA(
            attention_map.sigmoid(), layer1, layer2, layer3
        )

        layer1 = self.HL1(layer1)
        layer2 = self.HL2(layer2)
        layer3 = self.HL3_2(layer3)

        y = self.H2(layer3, layer2, layer1)  # *4

        y = self.output_head(y)  # 直接使用MCAP_Head

        return F.interpolate(
            attention_map, size=A.size()[2:], mode="bilinear", align_corners=True
        ), F.interpolate(y, size=A.size()[2:], mode="bilinear", align_corners=True)
