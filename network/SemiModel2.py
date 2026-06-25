# -------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import yaml

# -------------------------------------------------------------------------------


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


class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
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


class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
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


class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
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
            nn.Sigmoid()
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




class SemiModel(nn.Module):
    def __init__(
        self,
        cfg_path="/root/C2F-SemiCD-and-C2FNet-main/network/w48_256x256_adam_lr1e-3.yaml",
        channel=32,
    ):
        super(SemiModel, self).__init__()

        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        # ----------------- 替换VGG为HRNet -----------------
        self.hrnet = get_pose_net(cfg=config)  # 加载HRNet配置
        del self.hrnet.final_layer  # 移除原姿态估计输出层

        # ----------------- HRNet特征层适配 -----------------
        # HRNet四阶段输出通道配置（根据yaml文件）
        self.stage_channels = [48, 96, 192, 384]  # 对应STAGE4的NUM_CHANNELS--w48
        # 如果使用w32版本的HRNet，可以取消注释以下行
        # self.stage_channels = [32, 64, 128, 256]  # 对应STAGE4的NUM_CHANNELS--w32

        # ----------------- 调整后续模块输入通道 -----------------
        # GCM模块输入通道适配
        # GCM 模块输入通道适配
        self.rfb1 = GCM(self.stage_channels[0], channel)  # stage1: 48 -> 32
        self.rfb2 = GCM(self.stage_channels[1], channel)  # stage2: 96 -> 32
        self.rfb3 = GCM(self.stage_channels[2], channel)  # stage3: 192 -> 32
        self.rfb4 = GCM(self.stage_channels[3], channel)  # stage4: 384 -> 32
        self.rfb3_2 = GCM(channel, channel)  # 用于细化后的 stage3
        # 注意力模块通道适配
        self._rebuild_attention_modules()

        # ----------------- 特征融合模块保持结构 -----------------
        self.agg1 = aggregation_init(channel)
        self.agg2 = aggregation_final(channel)
        self.HA = Refine()

        # ----------------- 输出层调整 -----------------
        self.agant1 = self._make_agant_layer(3 * channel, 2 * channel)  # 输入通道减少

        # --------
        # self.agant2 = self._make_agant_layer(2 * channel, channel)
        # 使用GhostConv替代常规卷积
        self.agant2 = nn.Sequential(
            GhostConv(2 * channel, channel),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        # 添加ASPP模块
        self.aspp = ASPP(channel, channel)
        # ------

        self.out_conv = nn.Conv2d(channel, 1, kernel_size=1)

        # 添加特征交互模块
        self.feature_interaction1 = FeatureInteraction(self.stage_channels[0])
        self.feature_interaction2 = FeatureInteraction(self.stage_channels[1])
        self.feature_interaction3 = FeatureInteraction(self.stage_channels[2])
        self.feature_interaction4 = FeatureInteraction(self.stage_channels[3])
        # ----------------------------

    def _rebuild_attention_modules(self):
        """重建注意力模块适配HRNet通道"""
        # 通道注意力
        self.atten_A_channel_1 = ChannelAttention(self.stage_channels[0])
        self.atten_A_channel_2 = ChannelAttention(self.stage_channels[1])
        self.atten_A_channel_3 = ChannelAttention(self.stage_channels[2])
        self.atten_A_channel_4 = ChannelAttention(self.stage_channels[3])

        self.atten_B_channel_1 = ChannelAttention(self.stage_channels[0])
        self.atten_B_channel_2 = ChannelAttention(self.stage_channels[1])
        self.atten_B_channel_3 = ChannelAttention(self.stage_channels[2])
        self.atten_B_channel_4 = ChannelAttention(self.stage_channels[3])

        # 空间注意力（保持原结构）
        self.atten_A_spatial_1 = SpatialAttention()
        self.atten_A_spatial_2 = SpatialAttention()
        self.atten_A_spatial_3 = SpatialAttention()
        self.atten_A_spatial_4 = SpatialAttention()

        self.atten_B_spatial_1 = SpatialAttention()
        self.atten_B_spatial_2 = SpatialAttention()
        self.atten_B_spatial_3 = SpatialAttention()
        self.atten_B_spatial_4 = SpatialAttention()

        # 保存各阶段的配置
        self.stage2_cfg = self.hrnet.stage2_cfg
        self.stage3_cfg = self.hrnet.stage3_cfg
        self.stage4_cfg = self.hrnet.stage4_cfg

    def _get_hrnet_features(self, x):
        """获取HRNet四阶段所有分支特征"""
        # Stem层
        x = self.hrnet.conv1(x)
        x = self.hrnet.bn1(x)
        x = self.hrnet.relu(x)
        x = self.hrnet.conv2(x)
        x = self.hrnet.bn2(x)
        x = self.hrnet.relu(x)
        x = self.hrnet.layer1(x)

        # 存储所有阶段的分支特征
        all_features = []

        # 阶段1特征 (单分支)
        stage1_features = [x]
        all_features.append(stage1_features)  # [256通道]

        # 过渡层1 (256 -> [48, 96])
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.hrnet.transition1[i] is not None:
                x_list.append(self.hrnet.transition1[i](x))
            else:
                x_list.append(x)

        # Stage2 处理 (输出[48, 96]两个分支)
        stage2_features = self.hrnet.stage2(x_list)
        all_features.append(stage2_features)  # [48, 96]

        # 过渡层2 ([48, 96] -> [48, 96, 192])
        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if i < len(stage2_features):
                if self.hrnet.transition2[i] is not None:
                    x_list.append(self.hrnet.transition2[i](stage2_features[i]))
                else:
                    x_list.append(stage2_features[i])
            else:
                # 处理分支数不足的情况
                if self.hrnet.transition2[i] is not None:
                    x_list.append(self.hrnet.transition2[i](stage2_features[-1]))
                else:
                    x_list.append(stage2_features[-1])

        # Stage3 处理 (输出[48, 96, 192]三个分支)
        stage3_features = self.hrnet.stage3(x_list)
        all_features.append(stage3_features)  # [48, 96, 192]

        # 过渡层3 ([48, 96, 192] -> [48, 96, 192, 384])
        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if i < len(stage3_features):
                if self.hrnet.transition3[i] is not None:
                    x_list.append(self.hrnet.transition3[i](stage3_features[i]))
                else:
                    x_list.append(stage3_features[i])
            else:
                # 处理分支数不足的情况
                if self.hrnet.transition3[i] is not None:
                    x_list.append(self.hrnet.transition3[i](stage3_features[-1]))
                else:
                    x_list.append(stage3_features[-1])

        # print("x_list shapes:", [f.shape for f in x_list])
        # Stage4 处理 (输出[48, 96, 192, 384]四个分支)
        stage4_features = self.hrnet.stage4(x_list)
        all_features.append(stage4_features)  # [48, 96, 192, 384]

        return all_features

    def _adjust_feature_resolution(self, feature, target_size):
        """调整特征图分辨率到目标尺寸"""
        _, _, h, w = feature.size()
        if h != target_size[0] or w != target_size[1]:
            return F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=True
            )
        return feature

    def forward(self, A, B):
        # ----------------- 特征提取 -----------------

        # [torch.Size([1, 48, 64, 64]), torch.Size([1, 96, 32, 32]), torch.Size([1, 192, 16, 16]), torch.Size([1, 384, 8, 8])]
        # 获取 HRNet 特征
        A_features = self._get_hrnet_features(A)
        B_features = self._get_hrnet_features(B)

        # 提取每个阶段的最高分辨率特征
        # 原始分辨率: [64x64, 32x32, 16x16, 8x8]
        A0 = A_features[3][0]  # stage1: [B, 48, 64, 64]
        A1 = A_features[3][1]  # stage2: [B, 96, 32, 32]
        A2 = A_features[3][2]  # stage3: [B, 192, 16, 16]
        A3 = A_features[3][3]  # stage4: [B, 384, 8, 8]

        B0 = B_features[3][0]
        B1 = B_features[3][1]
        B2 = B_features[3][2]
        B3 = B_features[3][3]

        # 应用通道和空间注意力
        def apply_attention(feature, ch_attn, sp_attn):
            ch_att = ch_attn(feature)
            sp_att = sp_attn(feature)
            return feature * ch_att * sp_att

        # ------------------------------------------------------------------------------
        # 对 A 和 B 的每个特征应用注意力并融合
        layer1_A = apply_attention(A0, self.atten_A_channel_1, self.atten_A_spatial_1)
        layer1_B = apply_attention(B0, self.atten_B_channel_1, self.atten_B_spatial_1)

        layer2_A = apply_attention(A1, self.atten_A_channel_2, self.atten_A_spatial_2)
        layer2_B = apply_attention(B1, self.atten_B_channel_2, self.atten_B_spatial_2)

        layer3_A = apply_attention(A2, self.atten_A_channel_3, self.atten_A_spatial_3)
        layer3_B = apply_attention(B2, self.atten_B_channel_3, self.atten_B_spatial_3)

        layer4_A = apply_attention(A3, self.atten_A_channel_4, self.atten_A_spatial_4)
        layer4_B = apply_attention(B3, self.atten_B_channel_4, self.atten_B_spatial_4)

        # 使用特征交互模块
        diff1, sum1, cat1 = self.feature_interaction1(layer1_A, layer1_B)
        diff2, sum2, cat2 = self.feature_interaction2(layer2_A, layer2_B)
        diff3, sum3, cat3 = self.feature_interaction3(layer3_A, layer3_B)
        diff4, sum4, cat4 = self.feature_interaction4(layer4_A, layer4_B)

        # 使用差异特征作为主要变化信号，但保留一些共同特征
        alpha = 0.7  # 差异特征的权重
        layer1 = alpha * diff1 + (1 - alpha) * sum1
        layer2 = alpha * diff2 + (1 - alpha) * sum2
        layer3 = alpha * diff3 + (1 - alpha) * sum3
        layer4 = alpha * diff4 + (1 - alpha) * sum4
        # ------------------------------------------------------------------------------

        # 处理特征用于初始聚合 (使用 stage2,3,4 对应原始模型的 3,4,5)
        # 调整分辨率到 16x16 用于聚合

        # 通过 GCM 模块统一通道数
        layer2_gcm = self.rfb2(layer2)  # 96->32
        layer3_gcm = self.rfb3(layer3)  # 192->32
        layer4_gcm = self.rfb4(layer4)  # 384->32

        # 初始聚合生成注意力图
        attention_map = self.agg1(layer4_gcm, layer3_gcm, layer2_gcm)


        print(attention_map.shape)
        # 使用注意力图增强特征 (stage1,2,3)
        # 调整注意力图到各特征的分辨率
        attn_sigmoid = attention_map.sigmoid()
        attn_64x64 = F.interpolate(
            attn_sigmoid, size=(64, 64), mode="bilinear", align_corners=True
        )
        attn_32x32 = F.interpolate(
            attn_sigmoid, size=(32, 32), mode="bilinear", align_corners=True
        )
        attn_16x16 = F.interpolate(
            attn_sigmoid, size=(16, 16), mode="bilinear", align_corners=True
        )

        layer1_enhanced = layer1 + layer1 * attn_64x64
        layer2_enhanced = layer2 + layer2 * attn_32x32
        layer3_enhanced = layer3_gcm + layer3_gcm * attn_16x16

        # 细化增强后的特征
        layer1_refined = self.rfb1(layer1_enhanced)  # 48->32
        layer2_refined = self.rfb2(layer2_enhanced)  # 96->32
        layer3_refined = self.rfb3_2(layer3_enhanced)  # 192->32

        # 最终聚合
        y = self.agg2(layer3_refined, layer2_refined, layer1_refined)

        # ----------------- 最终输出 -----------------
        y = self.agant1(y)
        y = self.agant2(y)

        # ===== 添加ASPP =====
        y = self.aspp(y)
        # ===================

        y = self.out_conv(y)

        # 上采样到原始图像尺寸
        attention_map_upscaled = F.interpolate(
            attention_map, size=A.size()[2:], mode="bilinear", align_corners=True
        )
        y_upscaled = F.interpolate(
            y, size=A.size()[2:], mode="bilinear", align_corners=True
        )

        return attention_map_upscaled, y_upscaled

    def _make_agant_layer(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )


# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg["MODEL"]["EXTRA"]
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        self.stage3_cfg = extra["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        self.stage4_cfg = extra["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg["MODEL"]["NUM_JOINTS"],
            kernel_size=extra["FINAL_CONV_KERNEL"],
            stride=1,
            padding=1 if extra["FINAL_CONV_KERNEL"] == 3 else 0,
        )

        self.pretrained_layers = extra["PRETRAINED_LAYERS"]

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=""):
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ["bias"]:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info("=> loading pretrained model {}".format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if (
                    name.split(".")[0] in self.pretrained_layers
                    or self.pretrained_layers[0] == "*"
                ):
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error("=> please download pre-trained models first!")
            raise ValueError("{} is not exist!".format(pretrained))


def get_pose_net(cfg, is_train=True, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg["MODEL"]["INIT_WEIGHTS"]:
        model.init_weights(cfg["MODEL"]["PRETRAINED"])

    return model
