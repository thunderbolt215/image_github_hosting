import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.DDR_Encoder_arch import MSRResNet_noGR_fea

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DAM(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size):
        super(DAM, self).__init__()
        self.channels_out = out_nc
        self.channels_in = in_nc
        self.kernel_size = kernel_size

        self.sft = SFT_layer(self.channels_in, self.channels_out)
        self.SE = SEBlock(in_nc)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        sft_out = self.sft(x, inter)
        out = sft_out
        out = x + out
        
        # 在这里加入seblock
        out = self.SE(out)

        return out

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
    
        return x5 * 0.2 + x
    
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        
        self.DAM = DAM(nf, nf, 3)
        # 加一个自定义的权重参数
        self.weight = nn.Parameter(torch.tensor([0.2], requires_grad=True))

    def forward(self, x, inter):
        out = x
        
        out = self.RDB1(out)
        # out = self.DAM(out, inter)
        out = self.RDB2(out)
        # out = self.DAM(out, inter)
        out = self.RDB3(out)

        out = self.DAM(out, inter) * self.weight + out
        return out 

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, gc=32, scale=1):
        super(RRDBNet, self).__init__()
        # RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.nb = nb
        self.RRDB_block1 = RRDB(nf, gc)
        self.RRDB_block2 = RRDB(nf, gc)
        self.RRDB_block3 = RRDB(nf, gc)
        self.RRDB_block4 = RRDB(nf, gc)
        self.RRDB_block5 = RRDB(nf, gc)
        self.RRDB_block6 = RRDB(nf, gc)
        self.RRDB_block7 = RRDB(nf, gc)
        self.RRDB_block8 = RRDB(nf, gc)
        self.RRDB_block9 = RRDB(nf, gc)
        self.RRDB_block10 = RRDB(nf, gc)
        self.RRDB_block11 = RRDB(nf, gc)
        self.RRDB_block12 = RRDB(nf, gc)
        self.RRDB_block13 = RRDB(nf, gc)
        self.RRDB_block14 = RRDB(nf, gc)
        self.RRDB_block15 = RRDB(nf, gc)
        self.RRDB_block16 = RRDB(nf, gc)
        
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        # RRDB_block_f = RRDB
        # self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.scale = scale

    def forward(self, x, inter):
        fea = self.conv_first(x)
        # 这里改成了循环的方式，直接用RRDB_trunk会出问题
        trunk = fea
        trunk = self.RRDB_block1(trunk, inter)
        trunk = self.RRDB_block2(trunk, inter)
        trunk = self.RRDB_block3(trunk, inter)
        trunk = self.RRDB_block4(trunk, inter)
        trunk = self.RRDB_block5(trunk, inter)
        trunk = self.RRDB_block6(trunk, inter)
        trunk = self.RRDB_block7(trunk, inter)
        trunk = self.RRDB_block8(trunk, inter)
        trunk = self.RRDB_block9(trunk, inter)
        trunk = self.RRDB_block10(trunk, inter)
        trunk = self.RRDB_block11(trunk, inter)
        trunk = self.RRDB_block12(trunk, inter)
        trunk = self.RRDB_block13(trunk, inter)
        trunk = self.RRDB_block14(trunk, inter)
        trunk = self.RRDB_block15(trunk, inter)
        trunk = self.RRDB_block16(trunk, inter)
        # 要改成16个模块
        # for i in range(self.nb):
        #     trunk = self.RRDB_block(trunk, inter)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk

        if self.scale == 4:
          fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
          fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.scale == 1:
          fea = self.lrelu(self.upconv1(fea))
          fea = self.lrelu(self.upconv2(fea))
        
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class DDRlessnoise_RRDB_SFT_Attention(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=5, upscale=1):
        super(DDRlessnoise_RRDB_SFT_Attention, self).__init__()

        # Restorer
        self.R = RRDBNet(in_nc, out_nc, nf, nb, gc=32)
        
        # Encoder
        encoder = MSRResNet_noGR_fea(rfea_layer='RB16').cuda()
        encoder_pretrain_path = '../pretrained_models/DDR_Encoder/latest_G_lessnoise.pth'
        encoder.load_state_dict(torch.load(encoder_pretrain_path), strict=True)
        for p in encoder.parameters(): # 将需要冻结的参数的 requires_grad 设置为 False
            p.requires_grad = False
        self.E = encoder
        # encoder = encoder.eval()
        
        self.nf = nf

    def forward(self, x):
        
        _, inter = self.E(x) # [B, C, H, W]
        latent_code = inter#.reshape(x.shape[0], self.nf, x.shape[2], x.shape[3])
        restored = self.R(x, latent_code)
        return restored