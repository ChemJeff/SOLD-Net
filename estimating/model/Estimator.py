from torch import nn
from torch.nn.init import kaiming_normal_
import torch
import torch.nn.functional as F
import numpy as np
from .blocks import upconv, Conv2dBlock, ResBlocks, outputConv, HourGlass, ConvBlock

class FeatureBacknone(nn.Module):
    def __init__(self, n_downsample=3, n_res=2, input_dim=3, dim=64, norm='in', activ='relu', pad_type='zero'):
        super(FeatureBacknone, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class MaskDecoder(nn.Module):
    def __init__(self, n_upsample=3, n_res=2, dim=512, output_dim=1, res_norm='in', activ='relu', pad_type='zero'):
        super(MaskDecoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)]
        self.model += [nn.Sigmoid()]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class PosDecoder(nn.Module):
    def __init__(self, cin=512, activ='relu'):
        super(PosDecoder, self).__init__()
        self.cin = cin
        self.conv1 = Conv2dBlock(cin, 256, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 512, 30, 40 -> 256, 30, 40
        self.res_block1 = ResBlocks(1, 256, norm='in', activation=activ, pad_type='zero') # 256, 30, 40 -> 256, 30, 40
        self.pool1 = nn.AvgPool2d(2, stride=2) # 256, 30, 40 -> 256, 15, 20
        self.conv2 = Conv2dBlock(256, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 256, 15, 20 -> 128, 15, 20
        self.pool2 = nn.AvgPool2d(5, stride=5) # 128, 15, 30 -> 128, 3, 4
        self.fc = nn.Linear(1536, 256)

    def forward(self, input):
        out = self.conv1(input)
        out = self.res_block1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.view(-1, 1536)
        out = self.fc(out)
        out = torch.softmax(out, dim=1)
        out = out.view(-1, 1, 8, 32)
        return out

class SkyCodeEstimator(nn.Module):
    def __init__(self, cin=512, cout=16, activ='relu'):
        super(SkyCodeEstimator, self).__init__()
        self.cin = cin
        self.cout = cout
        self.conv1 = Conv2dBlock(cin, 256, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 512, 30, 40 -> 256, 30, 40
        self.res_block1 = ResBlocks(1, 256, norm='in', activation=activ, pad_type='zero') # 256, 30, 40 -> 256, 30, 40
        self.conv2 = Conv2dBlock(256, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 256, 30, 40 -> 128, 15, 20
        self.res_block2 = ResBlocks(1, 128, norm='in', activation=activ, pad_type='zero') # 128, 15, 20 -> 128, 15, 20
        self.conv3 = Conv2dBlock(128, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 128, 15, 20 -> 128, 8, 10
        self.res_block3 = ResBlocks(1, 128, norm='in', activation=activ, pad_type='zero') # 128, 8, 10 -> 128, 8, 10
        self.conv4 = Conv2dBlock(128, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 128, 8, 10 -> 64, 8, 10
        self.res_block4 = ResBlocks(1, 64, norm='none', activation=activ, pad_type='zero') # 64, 8, 10 -> 64, 8, 10
        self.conv5 = Conv2dBlock(64, 32, 3, 2, 1, norm='none', activation=activ, pad_type='zero') # 64, 8, 10 -> 32, 4, 5
        self.conv6 = Conv2dBlock(32, cout, 3, 1, 1, norm='none', activation='none', pad_type='zero') # 32, 4, 5 -> cout, 4, 5
        self.outpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        out = self.conv1(input)
        out = self.res_block1(out)
        out = self.conv2(out)
        out = self.res_block2(out)
        out = self.conv3(out)
        out = self.res_block3(out)
        out = self.conv4(out)
        out = self.res_block4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.outpool(out)
        out = out.view(-1, self.cout)
        return out


class SunCodeEstimator(nn.Module):
    def __init__(self, cin=512, cout=45, activ='relu'):
        super(SunCodeEstimator, self).__init__()
        self.cin = cin
        self.cout = cout
        self.conv1 = Conv2dBlock(cin, 256, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 512, 30, 40 -> 256, 30, 40
        self.res_block1 = ResBlocks(1, 256, norm='in', activation=activ, pad_type='zero') # 256, 30, 40 -> 256, 30, 40
        self.conv2 = Conv2dBlock(256, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 256, 30, 40 -> 128, 15, 20
        self.res_block2 = ResBlocks(1, 128, norm='in', activation=activ, pad_type='zero') # 128, 15, 20 -> 128, 15, 20
        self.conv3 = Conv2dBlock(128, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 128, 15, 20 -> 128, 8, 10
        self.res_block3 = ResBlocks(1, 128, norm='in', activation=activ, pad_type='zero') # 128, 8, 10 -> 128, 8, 10
        self.conv4 = Conv2dBlock(128, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 128, 8, 10 -> 64, 8, 10
        self.res_block4 = ResBlocks(1, 64, norm='none', activation=activ, pad_type='zero') # 64, 8, 10 -> 64, 8, 10
        self.conv5 = Conv2dBlock(64, 32, 3, 2, 1, norm='none', activation=activ, pad_type='zero') # 64, 8, 10 -> 32, 4, 5
        self.conv6 = Conv2dBlock(32, cout, 3, 1, 1, norm='none', activation='none', pad_type='zero') # 32, 4, 5 -> cout, 4, 5
        self.outpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        out = self.conv1(input)
        out = self.res_block1(out)
        out = self.conv2(out)
        out = self.res_block2(out)
        out = self.conv3(out)
        out = self.res_block3(out)
        out = self.conv4(out)
        out = self.res_block4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.outpool(out)
        out = out.view(-1, self.cout)
        return out

class LocalFeatureExtractor(nn.Module):
    def __init__(self, cin=512, cout=128, activ='relu'):
        super(LocalFeatureExtractor, self).__init__()
        self.cin = cin
        self.cout = cout
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(cin, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
            ) # 512, 30, 40 -> 256, 60, 80
        self.res_block1 = ResBlocks(1, 256, norm='in', activation=activ, pad_type='zero') # 256, 60, 80 -> 256, 60, 80
        self.hourglass1 = HourGlass(2, 256, 'group') # 256, 60, 80 -> 256, 60, 80
        self.conv_top1 = ConvBlock(256, 256, 'group') # 256, 60, 80 -> 256, 60, 80
        self.conv_last1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, 256),
            nn.ReLU()) # 256, 60, 80 -> 256, 60, 80
        self.conv_l1 = nn.Conv2d(256, cout, kernel_size=1, stride=1, padding=0) # 256, 60, 80 -> 128, 60, 80
        self.conv_al1 = nn.Conv2d(cout, 256, kernel_size=1, stride=1, padding=0) # 128, 60, 80 -> 256, 60, 80
        self.conv_bl1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0) # 256, 60, 80 -> 256, 60, 80
        self.hourglass2 = HourGlass(2, 256, 'group') # 256, 60, 80 -> 256, 60, 80
        self.conv_top2 = ConvBlock(256, 256, 'group') # 256, 60, 80 -> 256, 60, 80
        self.conv_last2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, 256),
            nn.ReLU()) # 256, 60, 80 -> 256, 60, 80
        self.conv_l2 = nn.Conv2d(256, cout, kernel_size=1, stride=1, padding=0) # 256, 60, 80 -> 128, 60, 80

    def forward(self, input):
        out = self.deconv1(input)
        input1 = self.res_block1(out)
        out1 = self.hourglass1(input1)
        out1_1 = self.conv_top1(out1)
        out1_2 = self.conv_l1(out1_1)
        out1_a = self.conv_al1(out1_2)
        out1_b = self.conv_bl1(out1_1)
        input2 = input1 + out1_a + out1_b
        out2 = self.hourglass2(input2)
        out2_1 = self.conv_top2(out2)
        out2_2 = self.conv_l2(out2_1)
        return [out1_2, out2_2]

class LocalCodeEstimator(nn.Module):
    def __init__(self, cin=1152, cout=64):
        super(LocalCodeEstimator, self).__init__()
        self.cin = cin
        self.cout = cout
        self.percp1 = nn.Sequential(nn.Linear(cin, 512), nn.LeakyReLU(0.1))
        self.percp2 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.1))
        self.percp3 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.1))
        self.percp4 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.1))

    def forward(self, input):
        input = input.view(-1, self.cin)
        out1 = self.percp1(input)
        out2 = self.percp2(out1)
        out3 = self.percp3(out2)
        out4 = self.percp4(out3)
        return out4

class SkyDecoder(nn.Module):
    def __init__(self, cin=64, cout=3, activ='relu'):
        super(SkyDecoder, self).__init__()
        self.cin = cin
        self.cout = cout
        self.fc = nn.Linear(cin, 512)
        self.conv1 = Conv2dBlock(8, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 4, 8, 8 -> 4, 8, 64
        self.conv2 = Conv2dBlock(64, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 8, 16, 64 -> 8, 16, 128
        self.conv3 = Conv2dBlock(128, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 16, 32, 128 -> 16, 32, 64
        self.conv4 = Conv2dBlock(64, 32, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 32, 64, 64 -> 32, 64, 32

        self.res_block1 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero') # 8, 16, 64 -> 8, 16, 64
        self.res_block2 = ResBlocks(2, 128, norm='in', activation=activ, pad_type='zero') # 16, 32, 128 -> 16, 32, 128
        self.res_block3 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero') # 32, 64, 64 -> 32, 64, 64
        self.res_block4 = ResBlocks(2, 32, norm='none', activation=activ, pad_type='zero') # 64, 128, 32 -> 64, 128, 32

        self.upconv1 = upconv(64, 64, k=3, stride=1, pad=1)
        self.upconv2 = upconv(128, 128, k=3, stride=1, pad=1) # 8, 32, 128 -> 16, 64, 128
        self.upconv3 = upconv(64, 64, k=3, stride=1, pad=1) # 16, 64, 64 -> 32, 128, 64
        
        self.outputconv_1 = Conv2dBlock(32, 16, 3, 1, 1, norm='none', activation=activ, pad_type='zero') # 64, 128, 32 -> 64, 128, 16
        self.outputconv_2 = outputConv(16, cout) # 64, 128, 16 -> 64, 128, 3

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        out = self.fc(input)
        out = out.view(-1, 8, 4, 16)
        out = self.conv1(out)
        out = self.res_block1(out)
        out = self.upconv1(out)
        out = self.conv2(out)
        out = self.res_block2(out)
        out = self.upconv2(out)
        out = self.conv3(out)
        out = self.res_block3(out)
        out = self.upconv3(out)
        out = self.conv4(out)
        out = self.res_block4(out)
        out = self.outputconv_1(out)
        out = self.outputconv_2(out)
        return out

class SunDecoder(nn.Module):
    def __init__(self, cin=64, cout=3, activ='relu'):
        super(SunDecoder, self).__init__()
        self.cin = cin
        self.cout = cout
        self.conv1 = Conv2dBlock(cin+1, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 32, 128, cout+1 -> 32, 128, 64
        self.res_block1 = ResBlocks(1, 64, norm='in', activation=activ) # 32, 128, 64 -> 32, 128, 64
        self.conv2 = Conv2dBlock(64, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 32, 128, 64 -> 32, 128, 128
        self.res_block2 = ResBlocks(1, 128, norm='in', activation=activ, pad_type='zero') # 32, 128, 128 -> 32, 128, 128
        self.conv3 = Conv2dBlock(128, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 32, 128, 128 -> 32, 128, 64
        self.res_block3 = ResBlocks(1, 64, norm='in', activation=activ, pad_type='zero') # 32, 128, 64 -> 32, 128, 64
        self.conv4 = Conv2dBlock(64, 32, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 32, 128, 64 -> 32, 128, 32
        self.res_block4 = ResBlocks(2, 32, norm='none', activation=activ, pad_type='zero') # 32, 128, 32 -> 32, 128, 32
        self.outputconv_1 = Conv2dBlock(32, 16, 3, 1, 1, norm='none', activation=activ, pad_type='zero') # 32, 128, 32 -> 32, 128, 16
        self.outputconv_2 = Conv2dBlock(16, cout, 3, 1, 1, norm='none', activation='none', pad_type='zero') # 32, 128, 16 -> 32, 128, 3

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, pos_mask):
        # pos_mask at 32*128 resolution
        input = input.view(-1, self.cin, 1, 1)
        input = input.repeat((1, 1, 32, 128))
        pos_mask = pos_mask.view(-1, 1, 32, 128)
        input = torch.cat([input, pos_mask], dim=1)
        out = self.conv1(input)
        out = self.res_block1(out)
        out = self.conv2(out)
        out = self.res_block2(out)
        out = self.conv3(out)
        out = self.res_block3(out)
        out = self.conv4(out)
        out = self.res_block4(out)
        out = self.outputconv_1(out)
        out = self.outputconv_2(out)
        return out

class LocalSilDecoder(nn.Module):
    def __init__(self, cin=64, cout=1, activ='relu') -> None:
        super(LocalSilDecoder, self).__init__()
        self.cin = cin
        self.cout = cout
        self.fc = nn.Linear(cin, 256)
        self.conv1 = Conv2dBlock(8, 32, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 4, 8, 8 -> 4, 8, 32
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest') # 4, 8, 32 -> 8, 16, 32
        self.res_block1 = ResBlocks(2, 32, norm='in', activation=activ, pad_type='zero') # 8, 16, 32 -> 8, 16, 32
        self.conv2 = Conv2dBlock(32, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 8, 16, 32 -> 8, 16, 64
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest') # 8, 16, 64 -> 16, 32, 64
        self.res_block2 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero') # 16, 32, 64 -> 16, 32, 64
        self.conv3 = Conv2dBlock(64, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 16, 32, 64 -> 16, 32, 64
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest') # 16, 32, 64 -> 32, 64, 64
        self.res_block3 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero') # 32, 64, 64 -> 32, 64, 64
        self.conv4 = Conv2dBlock(64, 32, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 32, 64, 64 -> 32, 64, 32
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest') # 32, 64, 32 -> 64, 128, 32
        self.res_block4 = ResBlocks(2, 32, norm='in', activation=activ, pad_type='zero') # 32, 128, 32 -> 32, 128, 32
        self.outputconv_1 = Conv2dBlock(32, 16, 3, 1, 1, norm='none', activation=activ, pad_type='zero') # 32, 128, 32 -> 32, 128, 16
        self.outputconv_2 = Conv2dBlock(16, cout, 3, 1, 1, norm='none', activation='none', pad_type='zero') # 32, 128, 16 -> 32, 128, 1
        self.activation = nn.Sigmoid()

    def forward(self, input):
        out = self.fc(input)
        out = out.view(-1, 8, 4, 8)
        out = self.conv1(out)
        out = self.up1(out)
        out = self.res_block1(out)
        out = self.conv2(out)
        out = self.up2(out)
        out = self.res_block2(out)
        out = self.conv3(out)
        out = self.up3(out)
        out = self.res_block3(out)
        out = self.conv4(out)
        out = self.up4(out)
        out = self.res_block4(out)
        out = self.outputconv_1(out)
        out = self.outputconv_2(out)
        out = self.activation(out)
        return out

class LocalAppSplitRenderer(nn.Module):
    def __init__(self, cin_l=64, cin_sky=64, cin_sun=64, cout=3, activ='relu'):
        super(LocalAppSplitRenderer, self).__init__()
        self.cin_l = cin_l
        self.cin_sky = cin_sky
        self.cin_sun = cin_sun
        self.cout = cout

        self.fc = nn.Linear(cin_l+cin_sky, 1024)
        self.conv1 = Conv2dBlock(8, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 8, 16, 8 -> 8, 16, 64
        self.conv2 = Conv2dBlock(64, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 16, 32, 64 -> 16, 32, 128
        self.conv3 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 32, 64, 128 -> 32, 64, 128
        self.conv4 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 64, 128, 128 -> 64, 128, 128
        self.conv5 = Conv2dBlock(128+cin_sun+1, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 64, 128, 128+cin_sun+1 -> 64, 128, 128
        self.conv6 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 64, 128, 128 -> 64, 128, 128
        self.conv7 = Conv2dBlock(128, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 64, 128, 128 -> 64, 128, 64
        self.conv8 = Conv2dBlock(64, 32, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 64, 128, 64 -> 64, 128, 32

        self.res_block1 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero') # 8, 16, 64 -> 8, 16, 64
        self.res_block2 = ResBlocks(2, 128, norm='in', activation=activ, pad_type='zero') # 16, 32, 128 -> 16, 32, 128
        self.res_block3 = ResBlocks(2, 128, norm='in', activation=activ, pad_type='zero') # 32, 64, 128 -> 32, 64, 128
        self.res_block4 = ResBlocks(2, 128, norm='none', activation=activ, pad_type='zero') # 64, 128, 128 -> 64, 128, 128
        self.res_block5 = ResBlocks(2, 128, norm='in', activation=activ) # 64, 128, 128 -> 64, 128, 64
        self.res_block6 = ResBlocks(2, 128, norm='in', activation=activ, pad_type='zero') # 64, 128, 128 -> 64, 128, 128
        self.res_block7 = ResBlocks(2, 64, norm='in', activation=activ, pad_type='zero') # 64, 128, 64 -> 64, 128, 64
        self.res_block8 = ResBlocks(2, 32, norm='in', activation=activ, pad_type='zero') # 64, 128, 32 -> 64, 128, 32

        self.upconv1 = upconv(64, 64, k=3, stride=1, pad=1) # 8, 16, 64 -> 16, 32, 64
        self.upconv2 = upconv(128, 128, k=3, stride=1, pad=1) # 16, 32, 128 -> 32, 64, 128
        self.upconv3 = upconv(128, 128, k=3, stride=1, pad=1) # 32, 64, 128 -> 64, 128, 128

        self.outputconv_1 = Conv2dBlock(32, 16, 3, 1, 1, norm='none', activation=activ, pad_type='zero') # 64, 128, 32 -> 64, 128, 16
        self.outputconv_2 = outputConv(16, cout) # 64, 128, 16 -> 64, 128, 3

    def forward(self, input_l, input_sky, input_sun, input_posmask):
        # pos_mask at 64*128 resolution
        input = torch.cat([input_l, input_sky], dim=1)
        out = self.fc(input)
        out = out.reshape(-1, 8, 8, 16)
        out = self.conv1(out)
        out = self.res_block1(out)
        out = self.upconv1(out)
        out = self.conv2(out)
        out = self.res_block2(out)
        out = self.upconv2(out)
        out = self.conv3(out)
        out = self.res_block3(out)
        out = self.upconv3(out)
        out = self.conv4(out)
        out = self.res_block4(out)
        # add global sun infomation
        input_sun = input_sun.view(-1, self.cin_sun, 1, 1)
        input_sun = input_sun.repeat((1, 1, 64, 128))
        pos_mask = input_posmask.view(-1, 1, 64, 128)
        out = torch.cat([out, input_sun, pos_mask], dim=1)
        out = self.conv5(out)
        out = self.res_block5(out)
        out = self.conv6(out)
        out = self.res_block6(out)
        out = self.conv7(out)
        out = self.res_block7(out)
        out = self.conv8(out)
        out = self.res_block8(out)
        out = self.outputconv_1(out)
        out = self.outputconv_2(out)
        return out

class SplitLightEstimator(nn.Module):
    def __init__(self):
        super(SplitLightEstimator, self).__init__()
        self.feat_exactor = FeatureBacknone(n_downsample=3, n_res=1, input_dim=3, dim=64, norm='in', activ='relu', pad_type='zero')
        self.mask_decoder = MaskDecoder(n_upsample=3, n_res=1, dim=512, output_dim=1, res_norm='in', activ='relu', pad_type='zero')
        self.pos_decoder = PosDecoder(cin=512, activ='relu')
        self.sky_estimator = SkyCodeEstimator(cin=512, cout=16, activ='relu')
        self.sun_estimator = SunCodeEstimator(cin=512, cout=45, activ='relu')
        self.local_extractor = LocalFeatureExtractor(cin=512, cout=128, activ='relu')
        self.local_estimator = LocalCodeEstimator(cin=1152, cout=64)
        self.sky_decoder = SkyDecoder(cin=16, cout=3, activ='relu')
        self.sun_decoder = SunDecoder(cin=45, cout=3, activ='relu')
        self.local_app_render = LocalAppSplitRenderer(cin_l=64, cin_sky=16, cin_sun=45, cout=3, activ='relu')
        self.local_sil_decoder = LocalSilDecoder(cin=64, cout=1, activ='relu')

    def forward_img(self, input_img):
        img_feat = self.feat_exactor(input_img)
        return img_feat

    def forward_global(self, feat):
        mask_est = self.mask_decoder(feat)
        pos_est = self.pos_decoder(feat)
        sky_code_est = self.sky_estimator(feat)
        sun_code_est = self.sun_estimator(feat)
        return [mask_est, pos_est, sky_code_est, sun_code_est]

    def get_local_feat(self, feat):
        local_feat_list = self.local_extractor(feat)
        return local_feat_list

    def get_patch_feat(self, local_feat, local_pos):
        patch_feat = []
        local_pos = local_pos // 4
        B, num_local, D = local_pos.shape
        for b in range(B):
            for lid in range(num_local):
                pos_left_ind = max(int(0), local_pos[b,lid,1]-1)
                pos_right_ind = min(int(80), pos_left_ind+3)
                pos_left_ind = pos_right_ind - 3
                pos_upper_ind = max(int(0), local_pos[b,lid,0]-3)
                pos_lower_ind = min(int(60), pos_upper_ind+3)
                pos_upper_ind = pos_lower_ind - 3
                patch_feat.append(local_feat[b,:,pos_upper_ind:pos_lower_ind, pos_left_ind:pos_right_ind]) # 128, 3, 3
        patch_feat = torch.stack(patch_feat, dim=0)
        patch_feat = patch_feat.view(B*num_local, -1)
        return patch_feat

    def forward_local(self, patch_feat):
        local_code_est = self.local_estimator(patch_feat)
        return local_code_est

    # When training/validating, using all-in-one `forward' function call
    def forward(self, input_img, local_pos=None, global_sky_code=None, global_sun_code=None, cosine_mask_fine=None, global_only=None):
        img_feat = self.forward_img(input_img)
        [mask_est, pos_est, sky_code_est, sun_code_est] = self.forward_global(img_feat)
        if global_only is not None:
            return mask_est, pos_est
        local_feat_list = self.get_local_feat(img_feat)
        local_feat = local_feat_list[-1]
        B, num_local, D = local_pos.shape
        patch_feat = self.get_patch_feat(local_feat, local_pos)
        local_code_est = self.forward_local(patch_feat)
        sky_est_raw = self.sky_decoder(sky_code_est)

        # use max confidence point as sun pos est and calculate cosine mask
        pos_est_conf = pos_est.clone().detach().view(B, -1) # B, 256
        max_idx = torch.argmax(pos_est_conf, dim=1) # B
        pos_est_conf[:, :] = 0.0
        pos_est_conf[torch.arange(B), max_idx] = 1.0
        pos_est_conf = pos_est_conf.view(B, 1, 8, 32) # B, 1, 8, 32
        idx_y, idx_x = np.unravel_index(max_idx.cpu().numpy(), (8, 32))
        azimuth_rad_est = (idx_x - 15.5)/16.0*np.pi # B
        elevation_rad_est = (7.5 - idx_y)/16.0*np.pi # B
        sun_unit_vec = np.array([np.cos(elevation_rad_est)*np.sin(azimuth_rad_est), # x
                                np.cos(elevation_rad_est)*np.cos(azimuth_rad_est), # y
                                np.sin(elevation_rad_est)]) # z 
        sun_unit_vec = sun_unit_vec.reshape(3, -1) # 3, B
        _tmp = np.mgrid[63:-1:-1,0:128:1]
        elevation_mask = _tmp[0][np.newaxis,:]
        azimuth_mask = _tmp[1][np.newaxis,:]
        elevation_mask = (elevation_mask - 31.5)/32*(np.pi/2) # 1, 64, 128
        azimuth_mask = (azimuth_mask - 63.5)/64*(np.pi) # 1, 64, 128
        unit_mask = np.stack([np.cos(elevation_mask)*np.sin(azimuth_mask),
                                np.cos(elevation_mask)*np.cos(azimuth_mask),
                                np.sin(elevation_mask)], axis=-1) # 1, 64, 128, 3
        cosine_mask_fine_est = -np.einsum('ijkl,lm->ijkm', unit_mask, sun_unit_vec) # 1, 64, 128, B
        cosine_mask_fine_est = np.clip(cosine_mask_fine_est, 0.0, 1.0).astype(np.float32)
        cosine_mask_fine_est = np.transpose(cosine_mask_fine_est, (3, 0, 1, 2)) # B, 1, 64, 128
        cosine_mask_fine_est = torch.tensor(cosine_mask_fine_est).to('cuda')
        sun_pos_y = (idx_y + 0.5) / 8.0 # B
        sun_pos_x = (idx_x + 0.5) / 32.0 # B
        sun_pos_y = np.clip(sun_pos_y, 0, 1)
        sun_pos_x = np.clip(sun_pos_x, 0, 1)
        pos_est_fine = np.zeros((B, 1, 32, 128), dtype=np.float32) # B, 1, 32, 128
        idx_y = np.clip(sun_pos_y*32, 0, 31.99).astype(int)
        idx_x = np.clip(sun_pos_x*128, 0, 127.99).astype(int)
        pos_left_ind = np.maximum(int(0), idx_x-3)
        pos_right_ind = np.minimum(int(128), pos_left_ind+8)
        pos_left_ind = pos_right_ind - 8
        pos_upper_ind = np.maximum(int(0), idx_y-3)
        pos_lower_ind = np.minimum(int(32), pos_upper_ind+8)
        pos_upper_ind = pos_lower_ind - 8
        for _i in range(B):
            pos_est_fine[_i, 0, pos_upper_ind[_i]:pos_lower_ind[_i], pos_left_ind[_i]:pos_right_ind[_i]] = 1.0
        pos_est_fine = torch.tensor(pos_est_fine).to('cuda')

        sun_est_raw = self.sun_decoder(sun_code_est, pos_est_fine)
        # if training, GT codes and cosine maks are used
        if global_sky_code is not None and global_sun_code is not None and cosine_mask_fine is not None:
            local_app_est_raw = self.local_app_render(local_code_est, global_sky_code.repeat_interleave(num_local, 0), global_sun_code.repeat_interleave(num_local, 0), cosine_mask_fine.repeat_interleave(num_local, 0))
        # if validating/testing, all codes and masks are predicted
        else:
            local_app_est_raw = self.local_app_render(local_code_est, sky_code_est.repeat_interleave(num_local, 0), sun_code_est.repeat_interleave(num_local, 0), cosine_mask_fine_est.repeat_interleave(num_local, 0))
        local_sil_est = self.local_sil_decoder(local_code_est)

        azimuth_deg_est = azimuth_rad_est / np.pi * 180.0
        elevation_deg_est = elevation_rad_est / np.pi * 180.0
        azimuth_deg_est = torch.tensor(azimuth_deg_est).to('cuda')
        elevation_deg_est = torch.tensor(elevation_deg_est).to('cuda')

        return mask_est, pos_est, sky_code_est, sun_code_est, local_code_est, sky_est_raw, sun_est_raw, local_app_est_raw, local_sil_est, azimuth_deg_est, elevation_deg_est, pos_est_fine, cosine_mask_fine_est