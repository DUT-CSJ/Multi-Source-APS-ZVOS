import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet101
from models.layer import FEM_channel, FEM_spatial, Decoder1, Decoder2
from models.stage1 import Stage1
import timm
from mmcls.apis import init_model


class Stage2(nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        #  resnet101 backbone  #
        resnet_flow = timm.create_model('resnet101d', pretrained=True)
        self.conv1_flow = resnet_flow.conv1
        self.bn1_flow = resnet_flow.bn1
        # self.relu_flow = resnet_flow.relu  # 1/2, 64
        self.relu_flow = resnet_flow.act1
        self.maxpool_flow = resnet_flow.maxpool
        self.conv2_flow = resnet_flow.layer1  # 1/4, 256
        self.conv3_flow = resnet_flow.layer2  # 1/8, 512
        self.conv4_flow = resnet_flow.layer3  # 1/16, 1024
        self.conv5_flow = resnet_flow.layer4  # 1/32, 2048
        #  Decoder flow Layer1 FPN  #
        self.T5_flow = resnet_flow.T5_flow
        self.T4_flow = resnet_flow.T4_flow
        self.T3_flow = resnet_flow.T3_flow
        self.T2_flow = resnet_flow.T2_flow
        #  Decoder rgb Layer1 FPN  #
        self.T5_rgb = Decoder1(2048, 256)
        self.T4_rgb = Decoder1(1024, 256)
        self.T3_rgb = Decoder1(512, 256)
        self.T2_rgb = Decoder1(256, 256)
        # self.T1_rgb = Decoder1(64, 64)
        #  Decoder flow Layer2 FPN  #
        self.P5_flow = resnet_flow.P5_flow
        self.P4_flow = resnet_flow.P4_flow
        self.P3_flow = resnet_flow.P3_flow
        self.P2_flow = resnet_flow.P2_flow
        #  Decoder rgb Layer2 FPN  #
        self.P5_rgb = Decoder2(256, 128)
        self.P4_rgb = Decoder2(256, 128)
        self.P3_rgb = Decoder2(256, 128)
        self.P2_rgb = Decoder2(256, 128)

        #  Fuse_5  #
        self.T5_fuse_initial = nn.Sequential(nn.Conv2d(128 * 4, 128 * 4, kernel_size=1), nn.BatchNorm2d(128 * 4),
                                             nn.GELU(), nn.Conv2d(128 * 4, 128 * 4, kernel_size=1),
                                             nn.BatchNorm2d(128 * 4), nn.GELU())
        self.T5_rgb_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T5_rgb_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T5_depth_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T5_depth_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T5_sal_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T5_sal_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T5_rgb_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        self.T5_depth_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T5_sal_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        # self.T5_fuse0 = nn.Sequential(nn.Conv2d(128 * 3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        #                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.T5_channel = FEM_spatial(128)
        self.T5_spatial = FEM_spatial()
        self.T5_attention = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.T5_temp_fuse = nn.Sequential(nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128 * 3),
                                          nn.GELU())
        self.T5_flow_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                             nn.GELU())

        self.T5_fuse_positive = nn.Sequential(nn.Conv2d(128 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                              nn.GELU())
        self.T5_fuse_negative = nn.Sequential(nn.Conv2d(128 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                              nn.GELU())
        self.T5_fuse = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.GELU())

        #  Fuse_4  #
        self.T4_fuse_initial = nn.Sequential(nn.Conv2d(128 * 4, 128 * 4, kernel_size=1), nn.BatchNorm2d(128 * 4),
                                             nn.GELU(), nn.Conv2d(128 * 4, 128 * 4, kernel_size=1),
                                             nn.BatchNorm2d(128 * 4), nn.GELU())
        self.T4_rgb_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T4_rgb_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T4_depth_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T4_depth_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T4_sal_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T4_sal_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T4_rgb_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        self.T4_depth_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T4_sal_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        # self.T4_fuse0 = nn.Sequential(nn.Conv2d(128 * 3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        #                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.T4_channel = FEM_spatial(128)
        self.T4_spatial = FEM_spatial()
        self.T4_attention = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.T4_temp_fuse = nn.Sequential(nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128 * 3),
                                          nn.GELU())
        self.T4_flow_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                             nn.GELU())

        self.T4_fuse_positive = nn.Sequential(nn.Conv2d(128 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                              nn.GELU())
        self.T4_fuse_negative = nn.Sequential(nn.Conv2d(128 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                              nn.GELU())
        self.T4_fuse = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.GELU())

        #  Fuse_3  #
        self.T3_fuse_initial = nn.Sequential(nn.Conv2d(128 * 4, 128 * 4, kernel_size=1), nn.BatchNorm2d(128 * 4),
                                             nn.GELU(), nn.Conv2d(128 * 4, 128 * 4, kernel_size=1),
                                             nn.BatchNorm2d(128 * 4), nn.GELU())
        self.T3_rgb_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T3_rgb_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T3_depth_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T3_depth_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T3_sal_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T3_sal_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T3_rgb_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        self.T3_depth_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T3_sal_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        # self.T3_fuse0 = nn.Sequential(nn.Conv2d(128 * 3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        #                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.T3_channel = FEM_spatial(128)
        self.T3_spatial = FEM_spatial()
        self.T3_attention = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.T3_temp_fuse = nn.Sequential(nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128 * 3),
                                          nn.GELU())
        self.T3_flow_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                             nn.GELU())

        self.T3_fuse_positive = nn.Sequential(nn.Conv2d(128 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                              nn.GELU())
        self.T3_fuse_negative = nn.Sequential(nn.Conv2d(128 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                              nn.GELU())
        self.T3_fuse = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.GELU())

        #  Fuse_2  #
        self.T2_fuse_initial = nn.Sequential(nn.Conv2d(128 * 4, 128 * 4, kernel_size=1), nn.BatchNorm2d(128 * 4),
                                             nn.GELU(), nn.Conv2d(128 * 4, 128 * 4, kernel_size=1),
                                             nn.BatchNorm2d(128 * 4), nn.GELU())
        self.T2_rgb_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T2_rgb_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T2_depth_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T2_depth_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T2_sal_sa = nn.Conv2d(128 * 4, 1, kernel_size=3, padding=1)
        self.T2_sal_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T2_rgb_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        self.T2_depth_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T2_sal_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                            nn.GELU())
        # self.T2_fuse0 = nn.Sequential(nn.Conv2d(128 * 3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        #                               nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.T2_channel = FEM_spatial(128)
        self.T2_spatial = FEM_spatial()
        self.T2_attention = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.T2_temp_fuse = nn.Sequential(nn.Conv2d(128 * 3, 128 * 3, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(128 * 3),
                                          nn.GELU())
        self.T2_flow_sa_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                             nn.GELU())

        self.T2_fuse_positive = nn.Sequential(nn.Conv2d(128 * 4, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                              nn.GELU())
        self.T2_fuse_negative = nn.Sequential(nn.Conv2d(128 * 4, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                              nn.GELU())
        self.T2_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.GELU())

        #  Fuse_1  #
        self.T1_fuse_initial = nn.Sequential(nn.Conv2d(64 * 2, 64 * 2, kernel_size=1), nn.BatchNorm2d(64 * 2),
                                             nn.GELU(), nn.Conv2d(64 * 2, 64 * 2, kernel_size=1),
                                             nn.BatchNorm2d(64 * 2), nn.GELU())
        self.T1_rgb_sa = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.T1_rgb_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T1_flow_sa = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.T1_flow_PPM_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)
        self.T1_rgb_sa_fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                            nn.GELU())
        self.T1_flow_sa_fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                             nn.GELU())
        self.T1_channel = FEM_spatial(64)
        self.T1_spatial = FEM_spatial()
        self.T1_attention = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.T1_temp_fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.T1_flow_sa_fuse2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                             nn.GELU())

        self.T1_fuse_positive = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T1_fuse_negative = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                              nn.GELU())
        self.T1_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU())

        #  output Layer  #
        self.output4_sal = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                         nn.GELU())
        self.output3_sal = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                         nn.GELU())
        self.output2_sal = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.output1_sal = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1))

    def forward(self, e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output5_d, output4_d, output3_d, output2_d, output1_d, output5_sal, output4_sal, output3_sal, output2_sal, output1_sal, flows):
        input = flows
        B, _, _, _ = input.size()
        #  resnet101 backbone  #
        x = self.conv1_flow(flows)
        x = self.bn1_flow(x)
        e1_flow = self.relu_flow(x)  # 1/2, 64
        x = self.maxpool_flow(e1_flow)  # 1/4, 64
        e2_flow = self.conv2_flow(x)
        e3_flow = self.conv3_flow(e2_flow)
        e4_flow = self.conv4_flow(e3_flow)
        e5_flow = self.conv5_flow(e4_flow)
        #  Decoder flow Layer1  #
        T5_flow = self.T5_flow(e5_flow)
        T4_flow = self.T4_flow(e4_flow)
        T3_flow = self.T3_flow(e3_flow)
        T2_flow = self.T2_flow(e2_flow)
        T1_flow = e1_flow
        #  Decoder rgb Layer2  #
        T5_rgb = self.T5_rgb(e5_rgb)
        T4_rgb = self.T4_rgb(e4_rgb)
        T3_rgb = self.T3_rgb(e3_rgb)
        T2_rgb = self.T2_rgb(e2_rgb)
        T1_rgb = e1_rgb
        #  Decoder flow Layer2  #
        P5_flow = self.P5_flow(T5_flow)
        P4_flow = self.P4_flow(T4_flow + F.interpolate(T5_flow, size=e4_rgb.size()[2:], mode='bilinear'))
        P3_flow = self.P3_flow(T3_flow + F.interpolate(T4_flow, size=e3_rgb.size()[2:], mode='bilinear'))
        P2_flow = self.P2_flow(T2_flow + F.interpolate(T3_flow, size=e2_rgb.size()[2:], mode='bilinear'))
        #  Decoder rgb Layer2  #
        P5_rgb = self.P5_rgb(T5_rgb)
        P4_rgb = self.P4_rgb(T4_rgb + F.interpolate(T5_rgb, size=e4_rgb.size()[2:], mode='bilinear'))
        P3_rgb = self.P3_rgb(T3_rgb + F.interpolate(T4_rgb, size=e3_rgb.size()[2:], mode='bilinear'))
        P2_rgb = self.P2_rgb(T2_rgb + F.interpolate(T3_rgb, size=e2_rgb.size()[2:], mode='bilinear'))
        #  Fuse_5  #
        T5_fuse_initial = self.T5_fuse_initial(torch.cat((P5_flow, P5_rgb, output5_d, output5_sal), 1))
        T5_rgb_sa = F.sigmoid(self.T5_rgb_PPM_fuse(self.PPM(self.T5_rgb_sa(T5_fuse_initial))))
        T5_rgb_sa_enhanced = self.T5_rgb_sa_fuse(P5_rgb * T5_rgb_sa) + P5_rgb
        T5_depth_sa = F.sigmoid(self.T5_depth_PPM_fuse(self.PPM(self.T5_depth_sa(T5_fuse_initial))))
        T5_depth_sa_enhanced = self.T5_depth_sa_fuse(output5_d * T5_depth_sa) + output5_d
        T5_sal_sa = F.sigmoid(self.T5_sal_PPM_fuse(self.PPM(self.T5_sal_sa(T5_fuse_initial))))
        T5_sal_sa_enhanced = self.T5_sal_sa_fuse(output5_sal * T5_sal_sa) + output5_sal

        temp = torch.cat((T5_rgb_sa_enhanced, T5_depth_sa_enhanced, T5_sal_sa_enhanced), 1)
        temp1 = P5_flow
        temp_ca = self.T5_channel(temp1)
        T5_attention = self.T5_spatial(temp_ca * temp1)
        T5_attention = self.T5_attention(torch.cat((T5_attention, F.interpolate(output1_sal, size=T5_attention.size()[2:], mode='bilinear', align_corners=True)), 1))
        temp = self.T5_temp_fuse(temp * T5_attention + temp)
        # temp = self.T5_temp_fuse(temp * T5_attention) + temp

        T5_flow_sa_enhanced = self.T5_flow_sa_fuse(P5_flow * T5_attention + P5_flow)
        # T5_flow_sa_enhanced = self.T5_flow_sa_fuse(P5_flow * T5_attention) + P5_flow
        T5_fuse_positive = self.T5_fuse_positive(torch.cat((temp, T5_flow_sa_enhanced), 1))
        T5_fuse_negative = self.T5_fuse_negative(torch.cat((temp, T5_flow_sa_enhanced), 1))
        T5_fuse = self.T5_fuse(T5_fuse_positive - T5_fuse_negative)
        T5_fuse_initial, T5_rgb_sa_enhanced, T5_depth_sa_enhanced, T5_sal_sa_enhanced, T5_flow_sa_enhanced, T5_fuse_positive, T5_fuse_negative = None, None, None, None, None, None, None
        #  Fuse_4  #
        T4_fuse_initial = self.T4_fuse_initial(torch.cat((P4_flow, P4_rgb, output4_d, output4_sal), 1))
        T4_rgb_sa = F.sigmoid(self.T4_rgb_PPM_fuse(self.PPM(self.T4_rgb_sa(T4_fuse_initial))))
        T4_rgb_sa_enhanced = self.T4_rgb_sa_fuse(P4_rgb * T4_rgb_sa) + P4_rgb
        T4_depth_sa = F.sigmoid(self.T4_depth_PPM_fuse(self.PPM(self.T4_depth_sa(T4_fuse_initial))))
        T4_depth_sa_enhanced = self.T4_depth_sa_fuse(output4_d * T4_depth_sa) + output4_d
        T4_sal_sa = F.sigmoid(self.T4_sal_PPM_fuse(self.PPM(self.T4_sal_sa(T4_fuse_initial))))
        T4_sal_sa_enhanced = self.T4_sal_sa_fuse(output4_sal * T4_sal_sa) + output4_sal

        temp = torch.cat((T4_rgb_sa_enhanced, T4_depth_sa_enhanced, T4_sal_sa_enhanced), 1)
        temp1 = P4_flow
        temp_ca = self.T4_channel(temp1)
        T4_attention = self.T4_spatial(temp_ca * temp1)
        T4_attention = self.T4_attention(
            torch.cat((T4_attention, F.interpolate(output1_sal, size=T4_attention.size()[2:], mode='bilinear', align_corners=True)), 1))
        temp = self.T4_temp_fuse(temp * T4_attention + temp)
        # temp = self.T4_temp_fuse(temp * T4_attention) + temp

        T4_flow_sa_enhanced = self.T4_flow_sa_fuse(P4_flow * T4_attention + P4_flow)
        # T4_flow_sa_enhanced = self.T4_flow_sa_fuse(P4_flow * T4_attention) + P4_flow
        T4_fuse_positive = self.T4_fuse_positive(torch.cat((temp, T4_flow_sa_enhanced), 1))
        T4_fuse_negative = self.T4_fuse_negative(torch.cat((temp, T4_flow_sa_enhanced), 1))
        T4_fuse = self.T4_fuse(T4_fuse_positive - T4_fuse_negative)
        T4_fuse_initial, T4_rgb_sa_enhanced, T4_depth_sa_enhanced, T4_sal_sa_enhanced, T4_flow_sa_enhanced, T4_fuse_positive, T4_fuse_negative = None, None, None, None, None, None, None
        #  Fuse_3  #
        T3_fuse_initial = self.T3_fuse_initial(torch.cat((P3_flow, P3_rgb, output3_d, output3_sal), 1))
        T3_rgb_sa = F.sigmoid(self.T3_rgb_PPM_fuse(self.PPM(self.T3_rgb_sa(T3_fuse_initial))))
        T3_rgb_sa_enhanced = self.T3_rgb_sa_fuse(P3_rgb * T3_rgb_sa) + P3_rgb
        T3_depth_sa = F.sigmoid(self.T3_depth_PPM_fuse(self.PPM(self.T3_depth_sa(T3_fuse_initial))))
        T3_depth_sa_enhanced = self.T3_depth_sa_fuse(output3_d * T3_depth_sa) + output3_d
        T3_sal_sa = F.sigmoid(self.T3_sal_PPM_fuse(self.PPM(self.T3_sal_sa(T3_fuse_initial))))
        T3_sal_sa_enhanced = self.T3_sal_sa_fuse(output3_sal * T3_sal_sa) + output3_sal

        temp = torch.cat((T3_rgb_sa_enhanced, T3_depth_sa_enhanced, T3_sal_sa_enhanced), 1)
        temp1 = P3_flow
        temp_ca = self.T3_channel(temp1)
        T3_attention = self.T3_spatial(temp_ca * temp1)
        T3_attention = self.T3_attention(
            torch.cat((T3_attention, F.interpolate(output1_sal, size=T3_attention.size()[2:], mode='bilinear', align_corners=True)), 1))
        temp = self.T3_temp_fuse(temp * T3_attention + temp)
        # temp = self.T3_temp_fuse(temp * T3_attention) + temp

        T3_flow_sa_enhanced = self.T3_flow_sa_fuse(P3_flow * T3_attention + P3_flow)
        # T3_flow_sa_enhanced = self.T3_flow_sa_fuse(P3_flow * T3_attention) + P3_flow
        T3_fuse_positive = self.T3_fuse_positive(torch.cat((temp, T3_flow_sa_enhanced), 1))
        T3_fuse_negative = self.T3_fuse_negative(torch.cat((temp, T3_flow_sa_enhanced), 1))
        T3_fuse = self.T3_fuse(T3_fuse_positive - T3_fuse_negative)
        T3_fuse_initial, T3_rgb_sa_enhanced, T3_depth_sa_enhanced, T3_sal_sa_enhanced, T3_flow_sa_enhanced, T3_fuse_positive, T3_fuse_negative = None, None, None, None, None, None, None
        #  Fuse_2  #
        T2_fuse_initial = self.T2_fuse_initial(torch.cat((P2_flow, P2_rgb, output2_d, output2_sal), 1))
        T2_rgb_sa = F.sigmoid(self.T2_rgb_PPM_fuse(self.PPM(self.T2_rgb_sa(T2_fuse_initial))))
        T2_rgb_sa_enhanced = self.T2_rgb_sa_fuse(P2_rgb * T2_rgb_sa) + P2_rgb
        T2_depth_sa = F.sigmoid(self.T2_depth_PPM_fuse(self.PPM(self.T2_depth_sa(T2_fuse_initial))))
        T2_depth_sa_enhanced = self.T2_depth_sa_fuse(output2_d * T2_depth_sa) + output2_d
        T2_sal_sa = F.sigmoid(self.T2_sal_PPM_fuse(self.PPM(self.T2_sal_sa(T2_fuse_initial))))
        T2_sal_sa_enhanced = self.T2_sal_sa_fuse(output2_sal * T2_sal_sa) + output2_sal

        temp = torch.cat((T2_rgb_sa_enhanced, T2_depth_sa_enhanced, T2_sal_sa_enhanced), 1)
        temp1 = P2_flow
        temp_ca = self.T2_channel(temp1)
        T2_attention = self.T2_spatial(temp_ca * temp1)
        T2_attention = self.T2_attention(
            torch.cat((T2_attention, F.interpolate(output1_sal, size=T2_attention.size()[2:], mode='bilinear', align_corners=True)), 1))
        temp = self.T2_temp_fuse(temp * T2_attention + temp)
        # temp = self.T2_temp_fuse(temp * T2_attention) + temp

        T2_flow_sa_enhanced = self.T2_flow_sa_fuse(P2_flow * T2_attention + P2_flow)
        # T2_flow_sa_enhanced = self.T2_flow_sa_fuse(P2_flow * T2_attention) + P2_flow
        T2_fuse_positive = self.T2_fuse_positive(torch.cat((temp, T2_flow_sa_enhanced), 1))
        T2_fuse_negative = self.T2_fuse_negative(torch.cat((temp, T2_flow_sa_enhanced), 1))
        T2_fuse = self.T2_fuse(T2_fuse_positive - T2_fuse_negative)
        # T2_fuse_initial, T2_rgb_sa_enhanced, T2_depth_sa_enhanced, T2_sal_sa_enhanced, T2_flow_sa_enhanced, T2_fuse_positive, T2_fuse_negative = None, None, None, None, None, None, None
        #  Fuse_1  #
        T1_fuse_initial = self.T1_fuse_initial(torch.cat((T1_rgb, T1_flow), 1))
        T1_rgb_sa = F.sigmoid(self.T1_rgb_PPM_fuse(self.PPM(self.T1_rgb_sa(T1_fuse_initial))))
        T1_rgb_sa_enhanced = self.T1_rgb_sa_fuse(T1_rgb * T1_rgb_sa) + T1_rgb
        T1_flow_sa = F.sigmoid(self.T1_flow_PPM_fuse(self.PPM(self.T1_flow_sa(T1_fuse_initial))))
        T1_flow_sa_enhanced = self.T1_flow_sa_fuse(T1_flow * T1_flow_sa) + T1_flow

        temp1 = T1_flow
        temp_ca = self.T1_channel(temp1)
        T1_attention = self.T1_spatial(temp_ca * temp1)
        T1_attention = self.T1_attention(
            torch.cat((T1_attention, F.interpolate(output1_sal, size=T1_attention.size()[2:], mode='bilinear')), 1))
        temp = self.T1_temp_fuse(T1_rgb_sa_enhanced * T1_attention + T1_rgb_sa_enhanced)
        # temp = self.T1_temp_fuse(T1_rgb_sa_enhanced * T1_attention) + T1_rgb_sa_enhanced

        T1_flow_sa_enhanced = self.T1_flow_sa_fuse2(T1_flow_sa_enhanced * T1_attention + T1_flow_sa_enhanced)
        # T1_flow_sa_enhanced = self.T1_flow_sa_fuse2(T1_flow_sa_enhanced * T1_attention) + T1_flow_sa_enhanced
        T1_fuse_positive = self.T1_fuse_positive(torch.cat((temp, T1_flow_sa_enhanced), 1))
        T1_fuse_negative = self.T1_fuse_negative(torch.cat((temp, T1_flow_sa_enhanced), 1))
        T1_fuse = self.T1_fuse(T1_fuse_positive - T1_fuse_negative)
        # T1_fuse_initial, T1_rgb_sa_enhanced, T1_depth_sa_enhanced, T1_sal_sa_enhanced, T1_flow_sa_enhanced, T1_fuse_positive, T1_fuse_negative = None, None, None, None, None, None, None

        #  output  #
        output4_sal_ = self.output4_sal(F.interpolate(T5_fuse, size=e4_rgb.size()[2:], mode='bilinear', align_corners=True) + T4_fuse)
        output3_sal_ = self.output3_sal(F.interpolate(output4_sal_, size=e3_rgb.size()[2:], mode='bilinear', align_corners=True) + T3_fuse)
        output2_sal_ = self.output2_sal(F.interpolate(output3_sal_, size=e2_rgb.size()[2:], mode='bilinear', align_corners=True) + T2_fuse)
        output1_sal_ = self.output1_sal(F.interpolate(output2_sal_, size=e1_rgb.size()[2:], mode='bilinear', align_corners=True) + T1_fuse)
        output1_sal = F.interpolate(output1_sal_, size=input.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return output1_sal, T5_attention, T4_attention, T3_attention, T2_attention, T1_attention
        # return torch.sigmoid(output1_sal), T5_attention, T4_attention, T3_attention, T2_attention, T1_attention
        return torch.sigmoid(output1_sal)

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
    def Normmaxmin(self, map):
        batch_size = map.size()[0]
        for i in range(batch_size):
            map[i,:,:,:] = (map[i,:,:,:] - map[i,:,:,:].min()) / (map[i,:,:,:].max() - map[i,:,:,:].min() + 1e-8)
        return map

if __name__ == "__main__":
    torch.cuda.set_device(0)
    model1 = Stage1().cuda()
    model1.eval()
    for param in model1.parameters():
        param.requires_grad = False
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384)).cuda()
    flow = torch.autograd.Variable(torch.randn(4, 3, 384, 384)).cuda()
    e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output5_d, output4_d, output3_d, output2_d, output1_d, output5_sal, output4_sal, output3_sal, output2_sal, output1_sal = model1(input)
    model2 = Stage2().cuda()
    output0= model2(e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output5_d, output4_d, output3_d, output2_d, output1_d, output5_sal, output4_sal, output3_sal, output2_sal, output1_sal, flow)

