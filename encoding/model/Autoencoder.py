import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from .blocks import upconv, Conv2dBlock, ResBlocks, outputConv


class LocalEncoder(nn.Module):
    def __init__(self, cin=3, cout=64, activ='relu'):
        super(LocalEncoder, self).__init__()
        self.cin = cin
        self.cout = cout
        self.conv1_1 = Conv2dBlock(cin, 32, 5, 1, 2, norm='in', activation=activ, pad_type='zero') # 64, 128, 3 -> 64, 128, 32
        self.res_block1 = ResBlocks(1, 32, norm='in', activation=activ, pad_type='zero') # 64, 128, 32 -> 64, 128, 32
        self.conv1_2 = Conv2dBlock(32, 64, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 64, 128, 32 -> 32, 64, 64
        self.conv2_1 = Conv2dBlock(64, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 32, 64, 64 -> 32, 64, 64
        self.res_block2 = ResBlocks(1, 64, norm='in', activation=activ, pad_type='zero') # 32, 64, 64 -> 32, 64, 64
        self.conv2_2 = Conv2dBlock(64, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 32, 64, 64 -> 16, 32, 128
        self.conv3_1 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 16, 32, 128 -> 16, 32, 128
        self.res_block3 = ResBlocks(1, 128, norm='in', activation=activ, pad_type='zero') # 16, 32, 128 -> 16, 32, 128
        self.conv3_2 = Conv2dBlock(128, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 16, 32, 128 -> 8, 16, 128
        self.conv4_1 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 8, 16, 128 -> 8, 16, 128
        self.res_block4 = ResBlocks(2, 128, norm='in', activation=activ, pad_type='zero') # 8, 16, 128 -> 8, 16, 128
        self.conv4_2 = Conv2dBlock(128, 64, 3, 2, 1, norm='none', activation=activ, pad_type='zero') # 8, 16, 128 -> 4, 8, 64
        self.conv5 = Conv2dBlock(64, cout, 3, 1, 1, norm='none', activation='none', pad_type='zero') # 4, 8, 64 -> 4, 8, cout
        self.outpool = nn.AdaptiveAvgPool2d((1, 1))

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
        out = self.conv1_1(input)
        out = self.res_block1(out)
        out = self.conv1_2(out)
        out = self.conv2_1(out)
        out = self.res_block2(out)
        out = self.conv2_2(out)
        out = self.conv3_1(out)
        out = self.res_block3(out)
        out = self.conv3_2(out)
        out = self.conv4_1(out)
        out = self.res_block4(out)
        out = self.conv4_2(out)
        out = self.conv5(out)
        out = self.outpool(out)
        out = out.view(-1, self.cout)
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

class LocalAppRenderer(nn.Module):
    def __init__(self, cin_l=64, cin_g=64, cout=3, activ='relu'):
        super(LocalAppRenderer, self).__init__()
        self.cin_l = cin_l
        self.cin_g = cin_g
        self.cout = cout
        self.fc = nn.Linear(cin_l+cin_g, 1024)
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

    def forward(self, input_l, input_g):
        input = torch.cat([input_l, input_g], dim=1)
        out = self.fc(input)
        out = out.view(-1, 8, 8, 16)
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

class GlobalEncoder(nn.Module):
    def __init__(self, cin=3, cout=64, activ='relu'):
        super(GlobalEncoder, self).__init__()
        self.cin = cin
        self.cout = cout
        self.conv1_1 = Conv2dBlock(cin, 32, 5, 1, 2, norm='in', activation=activ, pad_type='zero') # 32, 128, 3 -> 32, 128, 32
        self.res_block1 = ResBlocks(1, 32, norm='in', activation=activ, pad_type='zero') # 32, 128, 32 -> 32, 128, 32
        self.conv1_2 = Conv2dBlock(32, 64, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 32, 128, 32 -> 16, 64, 64
        self.conv2_1 = Conv2dBlock(64, 64, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 16, 64, 64 -> 16, 64, 64
        self.res_block2 = ResBlocks(1, 64, norm='in', activation=activ, pad_type='zero') # 16, 64, 64 -> 16, 64, 64
        self.conv2_2 = Conv2dBlock(64, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 16, 64, 64 -> 8, 32, 128
        self.conv3_1 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 8, 32, 128 -> 8, 32, 128
        self.res_block3 = ResBlocks(1, 128, norm='in', activation=activ, pad_type='zero') # 8, 32, 128 -> 8, 32, 128
        self.conv3_2 = Conv2dBlock(128, 128, 3, 2, 1, norm='in', activation=activ, pad_type='zero') # 8, 32, 128 -> 4, 16, 128
        self.conv4_1 = Conv2dBlock(128, 128, 3, 1, 1, norm='in', activation=activ, pad_type='zero') # 4, 16, 128 -> 4, 16, 128
        self.res_block4 = ResBlocks(2, 128, norm='none', activation=activ, pad_type='zero') # 4, 16, 128 -> 4, 16, 128
        self.conv4_2 = Conv2dBlock(128, 64, 3, 2, 1, norm='none', activation=activ, pad_type='zero') # 4, 16, 128 -> 2, 8, 64
        self.conv5 = Conv2dBlock(64, cout, 3, 1, 1, norm='none', activation='none', pad_type='zero') # 2, 8, 64 -> 2, 8, cout
        self.outpool = nn.AdaptiveAvgPool2d((1, 1))

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
        out = self.conv1_1(input)
        out = self.res_block1(out)
        out = self.conv1_2(out)
        out = self.conv2_1(out)
        out = self.res_block2(out)
        out = self.conv2_2(out)
        out = self.conv3_1(out)
        out = self.res_block3(out)
        out = self.conv3_2(out)
        out = self.conv4_1(out)
        out = self.res_block4(out)
        out = self.conv4_2(out)
        out = self.conv5(out)
        out = self.outpool(out)
        out = out.view(-1, self.cout)
        return out

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