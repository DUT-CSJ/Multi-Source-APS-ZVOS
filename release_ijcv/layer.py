import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from collections import OrderedDict


class Decoder1(nn.Module):
    def __init__(self, input_channel=2048, output_channel=256):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        self.net = nn.Sequential(nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, padding=0), nn.BatchNorm2d(self.out_channel), nn.GELU())

    def forward(self, x):
        x = self.net(x)
        return x


class Decoder2(nn.Module):
    def __init__(self, input_channel=256, output_channel=128):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        self.net = nn.Sequential(nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channel), nn.GELU(), nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channel), nn.GELU())

    def forward(self, x):
        x = self.net(x)
        return x


class ASPP(nn.Module): # deeplab

    def __init__(self, dim, in_dim=256):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.ReLU(inplace=True))
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
         )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim),nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3,conv4,conv5), 1))



class Decoder3(nn.Module):
    def __init__(self, input_channel=2048, output_channel=256):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        self.net = nn.Sequential(nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(self.out_channel), nn.GELU())

    def forward(self, x):
        x = self.net(x)
        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMLayer, self).__init__()
        # channel attention
        self.attn_init = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1), nn.BatchNorm2d(channel),
                                             nn.GELU(), nn.Conv2d(channel, channel, kernel_size=1),
                                             nn.BatchNorm2d(channel), nn.GELU())
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # spatial attention
        self.spatial_sa = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.spatial_conv = nn.Conv2d(4, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # channel attention
        b, c, _, _ = x.size()
        x_init = self.attn_init(x)
        y1 = self.channel_avg_pool(x_init).view(b, c)
        y2 = self.channel_max_pool(x_init).view(b, c)
        y = y1 + y2
        y = self.fc(y).view(b, c, 1, 1)
        z = x_init * y.expand_as(x)
        # spatial attention
        scale = F.sigmoid(self.spatial_conv(self.PPM(self.spatial_sa(z))))
        return x + z * scale

    def PPM(self, Pool_F):
        Pool_F2 = F.avg_pool2d(Pool_F, kernel_size=(2, 2))
        Pool_F4 = F.avg_pool2d(Pool_F, kernel_size=(4, 4))
        Pool_F6 = F.avg_pool2d(Pool_F, kernel_size=(6, 6))
        Pool_Fgolobal = F.adaptive_avg_pool2d(Pool_F, 1)
        fuse = torch.cat((F.interpolate(Pool_F2, size=Pool_F.size()[2:], mode='bilinear'),
                          F.interpolate(Pool_F4, size=Pool_F.size()[2:], mode='bilinear'),
                          F.interpolate(Pool_F6, size=Pool_F.size()[2:], mode='bilinear'),
                          F.interpolate(Pool_Fgolobal, size=Pool_F.size()[2:], mode='bilinear')), 1)
        return fuse


class FEM_channel(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(FEM_channel, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgpool = self.fc2(self.relu1(self.fc1(self.avg(x))))
        maxpool = self.fc2(self.relu1(self.fc1(self.max(x))))
        return self.sigmoid(avgpool + maxpool)


class FEM_spatial(nn.Module):
    def __init__(self, kernel_size=7):
        super(FEM_spatial, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((avg_pool, max_pool), dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PSPModule(nn.Module):
    def __init__(self, features=512, out_features=512, sizes=(1, 2, 4, 6), HW=192):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size, HW) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size, HW):
        prior = nn.AdaptiveAvgPool2d(output_size=(HW//size, HW//size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        set_priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        priors = set_priors + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)



class ConvNext(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ToQKV(nn.Module):
    """
        generate q, k, v from a Tensor x
    """
    def __init__(self, in_channel=128):
        super(ToQKV, self).__init__()
        self.in_channel = in_channel
        self.to_qk = nn.Conv2d(in_channel, in_channel // 4 * 2, kernel_size=1)
        self.to_v = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x):
        qkv = self.to_qk(x)
        q, k = torch.split(qkv, [self.in_channel // 4, self.in_channel // 4], dim=1)
        v = self.to_v(x)
        return q, k, v


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)
            out = out + ffn(out)
        return out


class MobileViT(nn.Module):
    def __init__(self, in_channel=512, dim=512, kernel_size=3, patch_size=6, heads=8, head_dim=64, mlp_dim=1024):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=1, heads=heads, head_dim=head_dim, mlp_dim=mlp_dim)

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        y = x.clone()  # bs,c,h,w

        ## Local Representation
        y = self.conv2(self.conv1(x))  # bs,dim,h,w

        ## Global Representation
        _, _, h, w = y.shape
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)  # bs,h,w,dim
        y = self.trans(y)
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph,
                      nw=w // self.pw)  # bs,dim,h,w

        ## Fusion
        y = self.conv3(y)  # bs,dim,h,w
        y = torch.cat([x, y], 1)  # bs,2*dim,h,w
        y = self.conv4(y)  # bs,c,h,w

        return y
