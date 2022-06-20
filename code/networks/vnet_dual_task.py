import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(1, 1, 3, padding=1)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # ---- reverse attention branch 5 ----
        self.ra5_conv = BasicConv3d(256, 256, kernel_size=1)
        self.ra5_up = UpsamplingDeconvBlock(256, 128, normalization=normalization)

        # ---- reverse attention branch 6 ----
        self.ra6_conv1 = BasicConv3d(256, 256, kernel_size=1)
        self.ra6_conv2 = BasicConv3d(256, 128, kernel_size=1)
        self.ra6_up = UpsamplingDeconvBlock(128, 64, normalization=normalization)

        # ---- reverse attention branch 7 ----
        self.ra7_conv1 = BasicConv3d(128,64, kernel_size=1)
        self.ra7_conv2 = BasicConv3d(64, 32, kernel_size=1)
        self.ra7_conv3 = BasicConv3d(32, 16, kernel_size=1)
        self.ra7_conv4 = BasicConv3d(16, 1, kernel_size=1)


    def encoder(self, input):
        x1 = self.block_one(input)  # (4,16,112,112,80)
        x1_dw = self.block_one_dw(x1)  # (4,32,56,56,40)

        x2 = self.block_two(x1_dw)  # (4,32,56,56,40)
        x2_dw = self.block_two_dw(x2)  # (4,64,28,28,20)

        x3 = self.block_three(x2_dw)  # (4,64,28,28,20)
        x3_dw = self.block_three_dw(x3)  # (4,128,14,14,10)

        x4 = self.block_four(x3_dw)  # (4,128,14,14,10)
        x4_dw = self.block_four_dw(x4)  # (4,256,7,7,5)

        x5 = self.block_five(x4_dw)  # (4,256,7,7,5)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)
        segmentation = self.out_conv(x9)  # (4,1,112,112,80)


        # ---- reverse attention branch_5 ----
        ra5 = -1 * (torch.sigmoid(x5)) + 1  # (4, 256, 7, 7, 5)
        x = F.relu(self.ra5_conv(ra5))  # (4, 256, 7, 7, 5)
        ra5_out = self.ra5_up(x)  # (4, 128, 14, 14, 10)

        # ---- reverse attention branch_6 ----
        ra6 = -1 * (torch.sigmoid(x6)) + 1  # (4, 128, 14, 14, 10)
        x = torch.cat((ra6, ra5_out), 1)   # (4, 256, 14, 14, 10)
        x = F.relu(self.ra6_conv1(x))  # (4, 256, 14, 14, 10)
        x = F.relu(self.ra6_conv2(x))  # (4, 128, 14, 14, 10)
        ra6_out = self.ra6_up(x)  # (4, 64, 28, 28, 20)

        # ---- reverse attention branch_7 ----
        ra7 = -1 * (torch.sigmoid(x7)) + 1  # (4, 64, 28, 28, 20)
        x = torch.cat((ra7, ra6_out), 1)   # (4, 128, 28, 28, 20)
        x = F.relu(self.ra7_conv1(x))  # (4, 64, 28, 28, 20)
        x = F.relu(self.ra7_conv2(x))  # (4, 32, 28, 28, 20)
        x = F.relu(self.ra7_conv3(x))  # (4, 16, 28, 28, 20)
        ra7_out = self.ra7_conv4(x)  # (4, 1, 28, 28, 20)

        lateral_map_7 = F.interpolate(ra7_out, scale_factor=4.0, mode='trilinear', align_corners=False)
        background = self.out_conv2(lateral_map_7)  # (4, 1, 112, 112, 80)

        return segmentation, background


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        segmentation, background = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return segmentation, background
