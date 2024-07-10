import torch
import torch.nn.functional as F
from torch import nn
from models.layer import Decoder1, Decoder2, ASPP
from torchvision.models.resnet import resnet101, ResNet101_Weights, resnet50
import timm


class Stage1(nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        #  ResNet101 backbone  #
        resnet_rgb = timm.create_model('resnet101d', pretrained=True)
        self.conv1_RGB = resnet_rgb.conv1
        self.bn1_RGB = resnet_rgb.bn1
        # self.relu_RGB = resnet_rgb.relu  # 1/2, 64
        self.relu_RGB = resnet_rgb.act1
        self.maxpool_RGB = resnet_rgb.maxpool
        self.conv2_RGB = resnet_rgb.layer1  # 1/4, 256
        self.conv3_RGB = resnet_rgb.layer2  # 1/8, 512
        self.conv4_RGB = resnet_rgb.layer3  # 1/16, 1024
        self.conv5_RGB = resnet_rgb.layer4  # 1/32, 2048

        #  Decoder depth Layer1 FPN  #
        self.T5_depth = Decoder1(2048, 256)
        # self.T5_depth = ASPP(2048, 256)
        self.T4_depth = Decoder1(1024, 256)
        self.T3_depth = Decoder1(512, 256)
        self.T2_depth = Decoder1(256, 256)
        # self.T2_depth2 = Decoder1(256, 64)
        self.T1_depth = Decoder1(64, 64)
        #  Decoder sal Layer1 FPN  #
        self.T5_sal = Decoder1(2048, 256)
        # self.T5_sal = ASPP(2048, 256)
        self.T4_sal = Decoder1(1024, 256)
        self.T3_sal = Decoder1(512, 256)
        self.T2_sal = Decoder1(256, 256)
        # self.T2_sal2 = Decoder1(256, 64)
        self.T1_sal = Decoder1(64, 64)

        #  Decoder depth Layer2 FPN  #
        self.P5_depth = Decoder2(256, 128)
        self.P4_depth = Decoder2(256, 128)
        self.P3_depth = Decoder2(256, 128)
        self.P2_depth = Decoder2(256, 128)
        # self.P1_depth = Decoder2(64, 64)
        #  Decoder sal Layer2 FPN  #
        self.P5_sal = Decoder2(256, 128)
        self.P4_sal = Decoder2(256, 128)
        self.P3_sal = Decoder2(256, 128)
        self.P2_sal = Decoder2(256, 128)
        # self.P1_sal = Decoder2(64, 64)
        #  Fuse_5  #
        self.T5_fuse_initial = nn.Sequential(nn.Conv2d(128 * 2, 128 * 2, kernel_size=1), nn.BatchNorm2d(128 * 2),
                                             nn.GELU(), nn.Conv2d(128 * 2, 128 * 2, kernel_size=1),
                                             nn.BatchNorm2d(128 * 2), nn.GELU())
        self.T5_depth_sa = nn.Conv2d(128 * 2, 1, kernel_size=3, padding=1)
        self.T5_depth_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T5_sal_sa = nn.Conv2d(128 * 2, 1, kernel_size=3, padding=1)
        self.T5_sal_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T5_depth_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T5_sal_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        #  Fuse_4  #
        self.T4_fuse_initial = nn.Sequential(nn.Conv2d(128 * 2, 128 * 2, kernel_size=1), nn.BatchNorm2d(128 * 2),
                                             nn.GELU(), nn.Conv2d(128 * 2, 128 * 2, kernel_size=1),
                                             nn.BatchNorm2d(128 * 2), nn.GELU())
        self.T4_depth_sa = nn.Conv2d(128 * 2, 1, kernel_size=3, padding=1)
        self.T4_depth_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T4_sal_sa = nn.Conv2d(128 * 2, 1, kernel_size=3, padding=1)
        self.T4_sal_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T4_depth_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T4_sal_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        #  Fuse_3  #
        self.T3_fuse_initial = nn.Sequential(nn.Conv2d(128 * 2, 128 * 2, kernel_size=1), nn.BatchNorm2d(128 * 2),
                                             nn.GELU(), nn.Conv2d(128 * 2, 128 * 2, kernel_size=1),
                                             nn.BatchNorm2d(128 * 2), nn.GELU())
        self.T3_depth_sa = nn.Conv2d(128 * 2, 1, kernel_size=3, padding=1)
        self.T3_depth_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T3_sal_sa = nn.Conv2d(128 * 2, 1, kernel_size=3, padding=1)
        self.T3_sal_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T3_depth_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T3_sal_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        #  Fuse_2  #
        self.T2_fuse_initial = nn.Sequential(nn.Conv2d(128 * 2, 128 * 2, kernel_size=1), nn.BatchNorm2d(128 * 2),
                                             nn.GELU(), nn.Conv2d(128 * 2, 128 * 2, kernel_size=1),
                                             nn.BatchNorm2d(128 * 2), nn.GELU())
        self.T2_depth_sa = nn.Conv2d(128 * 2, 1, kernel_size=3, padding=1)
        self.T2_depth_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T2_sal_sa = nn.Conv2d(128 * 2, 1, kernel_size=3, padding=1)
        self.T2_sal_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T2_depth_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T2_sal_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())

        #  output0 depth  #
        self.output0_depth = nn.Sequential(nn.Conv2d(128*4, 128*4, kernel_size=3, padding=1), nn.BatchNorm2d(128*4), nn.GELU(), nn.Conv2d(128*4, 1, kernel_size=3, padding=1))
        #  output0 sal  #
        self.output0_sal = nn.Sequential(nn.Conv2d(128*4, 128*4, kernel_size=3, padding=1), nn.BatchNorm2d(128*4), nn.GELU(), nn.Conv2d(128*4, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        #  Encoder  #
        x = self.conv1_RGB(x)
        x = self.bn1_RGB(x)
        e1_rgb = self.relu_RGB(x)
        x = self.maxpool_RGB(e1_rgb)
        e2_rgb = self.conv2_RGB(x)
        e3_rgb = self.conv3_RGB(e2_rgb)
        e4_rgb = self.conv4_RGB(e3_rgb)
        e5_rgb = self.conv5_RGB(e4_rgb)
        #  Decoder Depth Layer1  #
        T5_d = self.T5_depth(e5_rgb)
        T4_d = self.T4_depth(e4_rgb)
        T3_d = self.T3_depth(e3_rgb)
        T2_d = self.T2_depth(e2_rgb)
        T1_d = self.T1_depth(e1_rgb)
        #  Decoder sal Layer1  #
        T5_sal = self.T5_sal(e5_rgb)
        T4_sal = self.T4_sal(e4_rgb)
        T3_sal = self.T3_sal(e3_rgb)
        T2_sal = self.T2_sal(e2_rgb)
        T1_sal = self.T1_sal(e1_rgb)
        #  Decoder Depth Layer2  #
        P5_d = self.P5_depth(T5_d)
        P4_d = self.P4_depth(T4_d+F.interpolate(T5_d, size=e4_rgb.size()[2:], mode='bilinear'))
        P3_d = self.P3_depth(T3_d+F.interpolate(T4_d, size=e3_rgb.size()[2:], mode='bilinear'))
        P2_d = self.P2_depth(T2_d + F.interpolate(T3_d, size=e2_rgb.size()[2:], mode='bilinear'))
        # T2_d2 = self.T2_depth2(T2_d + F.interpolate(T3_d, size=e2_rgb.size()[2:], mode='bilinear'))
        # P1_d = self.P1_depth(T1_d+F.interpolate(T2_d2, size=e1_rgb.size()[2:], mode='bilinear'))
        #  Decoder sal Layer2  #
        P5_sal = self.P5_sal(T5_sal)
        P4_sal = self.P4_sal(T4_sal + F.interpolate(T5_sal, size=e4_rgb.size()[2:], mode='bilinear'))
        P3_sal = self.P3_sal(T3_sal + F.interpolate(T4_sal, size=e3_rgb.size()[2:], mode='bilinear'))
        P2_sal = self.P2_sal(T2_sal + F.interpolate(T3_sal, size=e2_rgb.size()[2:], mode='bilinear'))
        # T2_sal2 = self.T2_sal2(T2_sal + F.interpolate(T3_sal, size=e2_rgb.size()[2:], mode='bilinear'))
        # P1_sal = self.P1_sal(T1_sal + F.interpolate(T2_sal2, size=e1_rgb.size()[2:], mode='bilinear'))
        #  Fuse_5  #
        T5_fuse_initial = self.T5_fuse_initial(torch.cat((P5_d, P5_sal), 1))
        T5_depth_sa = torch.sigmoid(self.T5_depth_PPM_fuse(self.PPM(self.T5_depth_sa(T5_fuse_initial))))
        P5_d = self.T5_depth_sa_fuse(P5_d * T5_depth_sa) + P5_d
        T5_sal_sa = torch.sigmoid(self.T5_sal_PPM_fuse(self.PPM(self.T5_sal_sa(T5_fuse_initial))))
        P5_sal = self.T5_sal_sa_fuse(P5_sal * T5_sal_sa) + P5_sal
        #  Fuse_4  #
        T4_fuse_initial = self.T4_fuse_initial(torch.cat((P4_d, P4_sal), 1))
        T4_depth_sa = torch.sigmoid(self.T4_depth_PPM_fuse(self.PPM(self.T4_depth_sa(T4_fuse_initial))))
        P4_d = self.T4_depth_sa_fuse(P4_d * T4_depth_sa) + P4_d
        T4_sal_sa = torch.sigmoid(self.T4_sal_PPM_fuse(self.PPM(self.T4_sal_sa(T4_fuse_initial))))
        P4_sal = self.T4_sal_sa_fuse(P4_sal * T4_sal_sa) + P4_sal
        #  Fuse_3  #
        T3_fuse_initial = self.T3_fuse_initial(torch.cat((P3_d, P3_sal), 1))
        T3_depth_sa = torch.sigmoid(self.T3_depth_PPM_fuse(self.PPM(self.T3_depth_sa(T3_fuse_initial))))
        P3_d = self.T3_depth_sa_fuse(P3_d * T3_depth_sa) + P3_d
        T3_sal_sa = torch.sigmoid(self.T3_sal_PPM_fuse(self.PPM(self.T3_sal_sa(T3_fuse_initial))))
        P3_sal = self.T3_sal_sa_fuse(P3_sal * T3_sal_sa) + P3_sal
        #  Fuse_2  #
        T2_fuse_initial = self.T2_fuse_initial(torch.cat((P2_d, P2_sal), 1))
        T2_depth_sa = torch.sigmoid(self.T2_depth_PPM_fuse(self.PPM(self.T2_depth_sa(T2_fuse_initial))))
        P2_d = self.T2_depth_sa_fuse(P2_d * T2_depth_sa) + P2_d
        T2_sal_sa = torch.sigmoid(self.T2_sal_PPM_fuse(self.PPM(self.T2_sal_sa(T2_fuse_initial))))
        P2_sal = self.T2_sal_sa_fuse(P2_sal * T2_sal_sa) + P2_sal

        output5_d = P5_d
        output4_d = P4_d
        output3_d = P3_d
        output2_d = P2_d

        output5_sal = P5_sal
        output4_sal = P4_sal
        output3_sal = P3_sal
        output2_sal = P2_sal
        #  output0_depth  #
        output1_d = self.output0_depth(torch.concat((F.interpolate(output5_d, size=e2_rgb.size()[2:], mode='bilinear'),
                                                  F.interpolate(output4_d, size=e2_rgb.size()[2:], mode='bilinear'),
                                                  F.interpolate(output3_d, size=e2_rgb.size()[2:], mode='bilinear'),
                                                  F.interpolate(output2_d, size=e2_rgb.size()[2:], mode='bilinear')), 1))
        output1_d = F.interpolate(output1_d, size=input.size()[2:], mode='bilinear')
        #  output0_sal  #
        output1_sal = self.output0_sal(torch.concat((F.interpolate(output5_sal, size=e2_rgb.size()[2:], mode='bilinear'),
                                       F.interpolate(output4_sal, size=e2_rgb.size()[2:], mode='bilinear'),
                                       F.interpolate(output3_sal, size=e2_rgb.size()[2:], mode='bilinear'),
                                       F.interpolate(output2_sal, size=e2_rgb.size()[2:], mode='bilinear')), 1))
        output1_sal = F.interpolate(output1_sal, size=input.size()[2:], mode='bilinear')
        if self.training:
            return torch.sigmoid(output1_d),output1_sal
        # return e1_rgb,e2_rgb,e3_rgb,e4_rgb,e5_rgb, output5_d,output4_d, output3_d,output2_d,torch.sigmoid(output1_d),output5_sal, output4_sal, output3_sal, output2_sal,torch.sigmoid(output1_sal)
        return e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output5_d, output4_d, output3_d, output2_d, torch.sigmoid(output1_d), output5_sal, output4_sal, output3_sal, output2_sal, output1_sal

    def PPM(self, Pool_F):
        Pool_F2 = F.avg_pool2d(Pool_F, kernel_size=(2, 2))
        Pool_F4 = F.avg_pool2d(Pool_F, kernel_size=(4, 4))
        Pool_F6 = F.avg_pool2d(Pool_F, kernel_size=(6, 6))
        Pool_Fgolobal = F.adaptive_avg_pool2d(Pool_F, 1)
        fuse = torch.cat((Pool_F,
                          F.interpolate(Pool_F2, size=Pool_F.size()[2:], mode='bilinear'),
                          F.interpolate(Pool_F4, size=Pool_F.size()[2:], mode='bilinear'),
                          F.interpolate(Pool_F6, size=Pool_F.size()[2:], mode='bilinear'),
                          F.interpolate(Pool_Fgolobal, size=Pool_F.size()[2:], mode='bilinear')), 1)
        return fuse


if __name__ == "__main__":
    def cnn_paras_count(net):
        """cnn参数量统计, 使用方式cnn_paras_count(net)"""
        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in net.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        return total_params, total_trainable_params

    def get_model_para_number(model):
        total_number = 0
        learnable_number = 0
        for para in model.parameters():
            total_number += torch.numel(para)
            if para.requires_grad == True:
                learnable_number += torch.numel(para)
        return total_number, learnable_number

    model = Stage1().cuda()
    total_number, learnable_number = get_model_para_number(model)
    print(total_number, learnable_number)
