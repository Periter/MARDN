
from model import common
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import model.arch_util as mutil
def make_model(args, parent=False):
    return MARDN(args)


# branch fusion with channel-wise sub-network attention
class FusionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(1)
        # Squeeze and excitation
        self.se_x1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.se_x2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.se_x3 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        if len(x) == 2:
            avg_x1 = self.se_x1(self.avg_pool1(x[0]))
            avg_x2 = self.se_x2(self.avg_pool2(x[1]))
            attens = self.softmax(torch.cat([avg_x1, avg_x2], 2))
            y = attens[:, :, 0, :].unsqueeze(2) * x[0] + attens[:, :, 1, :].unsqueeze(2) * x[1]
        elif len(x) == 3:
            avg_x1 = self.se_x1(self.avg_pool1(x[0]))
            avg_x2 = self.se_x2(self.avg_pool2(x[1]))
            avg_x3 = self.se_x3(self.avg_pool3(x[2]))
            attens = self.softmax(torch.cat([avg_x1, avg_x2, avg_x3], 2))

            y = attens[:, :, 0, :].unsqueeze(2) * x[0] + attens[:, :, 1, :].unsqueeze(2) * x[1] + \
                attens[:, :, 2, :].unsqueeze(2) * x[2]
        return y

class SpatialAttention(nn.Module):
    def __init__(self, n_feas):
        super(SpatialAttention, self).__init__()
        self.con1x1_pool = nn.Conv2d(n_feas, 1, 1)
        self.con1 = nn.Conv2d(3, 1, 1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        con_out = self.con1x1_pool(x)
        y = self.sig(self.con1(torch.cat([avg_out, max_out, con_out], 1)))

        return x*y

class ARDU(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ARDU, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.spatialAttention = SpatialAttention(nf)
        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.spatialAttention(x5)
        return x5 + x


class ARDB(nn.Module):
    '''Space-attended Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(ARDB, self).__init__()
        self.RDB1 = ARDU(nf, gc)
        self.RDB2 = ARDU(nf, gc)
        self.RDB3 = ARDU(nf, gc)
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class MARDN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MARDN, self).__init__()
        n_rrdb_b1 = args.n_level1
        n_rrdb_b2 = args.n_level2
        n_rrdb_b3 = args.n_level3
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        # module body
        # Each branch represents a sub-network
        ARDB_block_f = functools.partial(ARDB, nf=n_feats)
        # branch I
        modules_b1s1 = []
        modules_b1s2 = []
        modules_b1s3 = []
        for _ in range(n_rrdb_b1):
            modules_b1s1.append(ARDB_block_f())
            modules_b1s2.append(ARDB_block_f())
            modules_b1s3.append(ARDB_block_f())
        # branch II
        modules_b2s2 = []
        modules_b2s3 = []
        for _ in range(n_rrdb_b2):
            modules_b2s2.append(ARDB_block_f())
            modules_b2s3.append(ARDB_block_f())
        # branch III
        modules_b3s3 = []
        for _ in range(n_rrdb_b3):
            modules_b3s3.append(ARDB_block_f())


        # fusion layer
        self.furLayerb1s3 = FusionLayer(n_feats)
        self.furLayerb2s2 = FusionLayer(n_feats)
        self.furLayerb2s3 = FusionLayer(n_feats)
        self.furLayerb3s3 = FusionLayer(n_feats)
        self.furLayerFin = FusionLayer(n_feats)
        modules_tail = [
            conv(n_feats, args.n_colors, kernel_size)]
        modules_tail_1dot5 = [
            conv(n_feats, n_feats, kernel_size),
            conv(n_feats, args.n_colors, kernel_size)]
        modules_tailx2 = [
            common.Upsampler(conv, 2, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        modules_tailx4 = [
            common.Upsampler(conv, 4, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1_3 = nn.Sequential(
            nn.Conv2d(n_feats * 3, n_feats, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(*modules_head)
        self.modules_b1s1 = nn.Sequential(*modules_b1s1)
        self.modules_b1s2 = nn.Sequential(*modules_b1s2)
        self.modules_b1s3 = nn.Sequential(*modules_b1s3)
        self.modules_b2s2 = nn.Sequential(*modules_b2s2)
        self.modules_b2s3 = nn.Sequential(*modules_b2s3)
        self.modules_b3s3 = nn.Sequential(*modules_b3s3)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        # Upsample modules
        self.tail = nn.Sequential(*modules_tail)
        self.tail_1dot5 = nn.Sequential(*modules_tail_1dot5)
        self.tailx2 = nn.Sequential(*modules_tailx2)
        self.tailx4 = nn.Sequential(*modules_tailx4)
    def forward(self, x):
        x = self.sub_mean(x)
        # shallow feature extraction module
        x = self.head(x)
        # deep feature extraction module
        xh, xw = x.size(2), x.size(3)
        x_res = F.interpolate(x, size=(xh * 2, xw * 2), mode='bilinear', align_corners=False)  # bilinear
        # Stage I
        x_b1 = F.interpolate(x, size=(xh//2, xw//2), mode='bilinear', align_corners=False)   # bilinear
        x_b1 = self.modules_b1s1(x_b1)
        # Stage II
        x_b2 = self.furLayerb2s2([F.interpolate(x_b1, size=(xh, xw), mode='bilinear', align_corners=False), x])
        x_b2 = self.modules_b2s2(x_b2)
        x_b1 = self.modules_b1s2(x_b1)
        # Stage III
        x_b3 = self.furLayerb3s3([F.interpolate(x_b1, size=(xh*2, xw*2), mode='bilinear', align_corners=False),\
                                  F.interpolate(x_b2, size=(xh*2, xw*2), mode='bilinear', align_corners=False)])
        x_temp = self.furLayerb1s3([x_b1, F.interpolate(x_b2, size=(xh//2, xw//2), mode='bilinear', align_corners=False)])
        x_b2 = self.furLayerb2s3([F.interpolate(x_b1, size=(xh, xw), mode='bilinear', align_corners=False), x_b2])
        x_b1 = x_temp
        x_b3 = self.modules_b3s3(x_b3)
        x_b2 = self.modules_b2s3(x_b2)
        x_b1 = self.modules_b1s3(x_b1)

        res = self.furLayerFin([F.interpolate(x_b1, size=(2*xh, 2*xw), mode='bilinear', align_corners=False),\
                                F.interpolate(x_b2, size=(2*xh, 2*xw), mode='bilinear', align_corners=False), x_b3])
        res = res + x_res
        # Upscaling and reconstruct module
        if self.scale == 2:
            x = self.tail(res)
        elif self.scale == 3:
            x = F.interpolate(res, size=(3*xh, 3*xw), mode='bilinear', align_corners=False)
            x = self.tail_1dot5(x)
        elif self.scale == 4:
            x = self.tailx2(res)
        elif self.scale == 8:
            x = self.tailx4(res)
        x = self.add_mean(x)

        return x