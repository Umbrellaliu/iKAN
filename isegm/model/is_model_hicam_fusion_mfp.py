import torch
import torch.nn as nn
import numpy as np

from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from isegm.model.modifiers import LRMult
import time
import torch
import torch.nn as nn
import numpy as np

from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from isegm.model.modifiers import LRMult
from isegm.model.modulation import *

import cv2

import torch
import torch.nn as nn


class Inc(nn.Module):
    def __init__(self, in_channels, filters):
        super(Inc, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1,
                      padding=(1 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(3, 3), stride=(1, 1), dilation=1,
                      padding=(3 - 1) // 2),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1,
                      padding=(1 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(5, 5), stride=(1, 1), dilation=1,
                      padding=(5 - 1) // 2),
            nn.LeakyReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1),
            nn.LeakyReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1),
            nn.LeakyReLU(),
        )

    def forward(self, input):
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        return torch.cat([o1, o2, o3, o4], dim=1)

class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Softsign()
        )

    def forward(self, input):
        return input * self.module(input).unsqueeze(2).unsqueeze(3).expand_as(input)

class hicam(nn.Module):
    def __init__(self):
        super(hicam, self).__init__()
        self.layer_1 = CAM(6, 1)
        self.layer_2= Inc(in_channels=6, filters=64)
        # self.layer_1_mfp = CAM(7, 1)
        # self.layer_2_mfp= Inc(in_channels=7, filters=64)
        self.layer_3= CAM(256,4)

        # 输出层
        self.layer_tail = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(3 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(1 - 1) // 2),
            nn.Sigmoid()
            #nn.LeakyReLU(),
        )

    def forward(self, image, coord_features, prev_mask_modulated):

        layer_concat = torch.cat([image, coord_features,prev_mask_modulated], dim=1)
        layer_concat_main = torch.cat([image, coord_features], dim=1)
        layer_1 = self.layer_1(layer_concat_main)
        layer_2 = self.layer_2(layer_1)
        layer_3 = self.layer_3(layer_2)

        # layer_1_mfp = self.layer_1_mfp(layer_concat)
        # layer_2_mfp = self.layer_2_mfp(layer_1_mfp)
        # layer_3_mfp = self.layer_3(layer_2_mfp)

        output = self.layer_tail(layer_3)
        # output_mfp = self.layer_tail(layer_3_mfp)
        return output,layer_concat

class ISModel_hicam_fusion_mfp(nn.Module):
    def __init__(self, use_rgb_conv=True, with_aux_output=False,
                 norm_radius=260, use_disks=False, cpu_dist_maps=False,
                 clicks_groups=None, with_prev_mask=True, use_leaky_relu=False,
                 binary_prev_mask=False, conv_extend=False, norm_layer=nn.BatchNorm2d,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225]), use_hicam=True,N=100, R_max=25):
        """
        初始化ISModel类，支持多种配置（如是否启用hicam、是否使用RGB卷积、是否包含辅助输出等）。
        """
        super().__init__()

        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.N=N
        self.R_max=R_max

        # 初始化hicam模块
        self.use_hicam = use_hicam
        if self.use_hicam:
            self.hicam_module = hicam()  # 使用您之前定义好的hicam模块

        # 其他卷积和特征处理层
        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask:
            self.coord_feature_ch += 1

        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv2d(in_channels=3 + self.coord_feature_ch, out_channels=6 + self.coord_feature_ch, kernel_size=1),
                norm_layer(6 + self.coord_feature_ch),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6 + self.coord_feature_ch, out_channels=3, kernel_size=1)
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=64,
                                            kernel_size=3, stride=2, padding=1)
            self.maps_transform.apply(LRMult(0.1))
        else:
            self.rgb_conv = None
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)

        # dist_maps 初始化
        if self.clicks_groups is not None:
            self.dist_maps = nn.ModuleList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(DistMaps(norm_radius=click_radius, spatial_scale=1.0,
                                               cpu_mode=cpu_dist_maps, use_disks=use_disks))
        else:
            self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                      cpu_mode=cpu_dist_maps, use_disks=use_disks)

    def forward(self, image, points):
        """
        前向传播方法，支持通过传递 `image` 和 `points` 来进行分割计算。
        如果启用了hicam模块，则会使用hicam进行处理。
        """
        # 准备输入图像，处理 prev_mask 和 coord_features
        # 准备输入图像，处理 prev_mask 和 coord_features
        image, prev_mask, prev_mask_modulated = self.prepare_input(image)
        if prev_mask_modulated.shape[1] == 0:
            points = points.float()
            prev_mask_modulated = modulate_prevMask(prev_mask, points, self.N, self.R_max)

        coord_features = self.get_coord_features(image, prev_mask, points)

        # 如果启用了hicam模块，使用hicam处理图像
        if self.use_hicam:
            # 传递 `image`, `coord_features` 和 `prev_mask` 到 hicam 模块
            x,layer_1 = self.hicam_module(image, coord_features, prev_mask_modulated)
            outputs= self.backbone_forward(x,layer_1)


        else:
            # 如果没有启用 hicam，使用普通的特征提取网络
            if self.rgb_conv is not None:
                x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
                layer_concat = torch.cat([image, coord_features,prev_mask_modulated], dim=1)
                outputs = self.backbone_forward(x,layer_concat)
            else:
                coord_features = self.maps_transform(coord_features)
                outputs = self.backbone_forward(image, coord_features)

        # 对 `instances` 进行插值，确保输出大小与输入一致
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)

        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                                 mode='bilinear', align_corners=True)

        return outputs

    def backbone_forward(self, image, every_features):
        """
        假设这是一个占位函数，您可以根据实际的网络架构（如ResNet, UNet等）进行实现。
        """
        raise NotImplementedError

    def prepare_input(self, image):
        """
        准备输入图像，进行归一化并处理 prev_mask（如果启用）。
        """
        #print(image.shape)
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:4, :, :]  # 假设 prev_mask 位于输入图像的3个通道后
            prev_mask_modulated = image[:, 4:, :, :]
            # print(1,prev_mask.shape)
            # print(2,prev_mask_modulated.shape)
            image = image[:, :3, :, :]  # 保留图像的前3个通道（RGB）

            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()  # 如果是二值化的前景掩码

        image = self.normalization(image)
        #print(image.shape)
        return image, prev_mask, prev_mask_modulated

    def get_coord_features(self, image, prev_mask, points):
        """
        获取交互特征（如距离图），并根据需要将 prev_mask 拼接进来。
        """
        if self.clicks_groups is not None:
            points_groups = split_points_by_order(points, groups=(2,) + (1,) * (len(self.clicks_groups) - 2) + (-1,))
            coord_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
            coord_features = torch.cat(coord_features, dim=1)
        else:
            coord_features = self.dist_maps(image, points)

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features

def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points

