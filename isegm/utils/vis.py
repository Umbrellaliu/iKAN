from functools import lru_cache

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_instances(imask, bg_color=0,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def visualize_mask(mask, num_cls):
    palette = get_palette(num_cls)
    mask[mask == -1] = 0

    return palette[mask].astype(np.uint8)


def visualize_proposals(proposals_info, point_color=(255, 0, 0), point_radius=1):
    proposal_map, colors, candidates = proposals_info

    proposal_map = draw_probmap(proposal_map)
    for x, y in candidates:
        proposal_map = cv2.circle(proposal_map, (y, x), point_radius, point_color, -1)

    return proposal_map


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def add_tag(image, tag = 'nodefined', tag_h = 40):
    image = image.astype(np.uint8)
    H,W = image.shape[0], image.shape[1]
    tag_blanc = np.ones((tag_h,W,3)).astype(np.uint8) * 255
    cv2.putText(tag_blanc,tag,(10,30),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0 ), 1)
    #print(image.shape,tag_blanc.shape)
    image = cv2.vconcat([image,tag_blanc])
    return image
def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=np.bool_)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(np.bool_)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries
    
 
def draw_with_blend_and_clicks(img, mask=None, alpha=0.3, clicks_list=None, pos_color=(0, 255, 0),
                               neg_color=(255, 0, 0), radius=4):
    result = img.copy()

    if mask is not None:

        palette = get_palette(np.max(mask) + 1)
        palette = np.zeros((2, 3), dtype=np.uint8)
        palette[1:] = [0, 255, 0]  # 所有前景类别为绿色
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result


def display_images(additional_image=None,every_features=None):
    with (torch.no_grad()):
        # 每个样本的数量
        every_features = every_features.cpu().numpy()

        num_samples = every_features.shape[0]

        # 创建一个fig，设置每一行显示三个图像（RGB图像 + 点图像 + 灰度图像）
        #fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

        # 如果是单张图像，调整axs的形状为二维数组
        batch_size, channels, height, width = every_features.shape

        # 创建子图
        fig, axs = plt.subplots(batch_size, 4, figsize=(12, 4 * batch_size))
        if num_samples == 1:
            axs = np.expand_dims(axs, axis=0)

        for idx in range(batch_size):
            sample = every_features[idx]  # 选择每个样本

            # 提取 RGB 图像
            rgb_image = sample[:3]  # 前三个通道是RGB
            rgb_image = np.transpose(rgb_image, (1, 2, 0))  # 转置为 (height, width, channels)
            rgb_image = (rgb_image * 255).astype(np.uint8)  # 转换为 [0, 255] 范围

            # 获取第5和第6通道，作为红色和绿色点的位置
            red_points = sample[4]  # 第5通道作为红色点
            green_points = sample[5]  # 第6通道作为绿色点
            red_points = (red_points!=0 ).astype(np.uint8)  # 简单阈值化处理
            green_points = (green_points !=0).astype(np.uint8)  # 简单阈值化处理

            # 提取灰度图像
            gray_image1 = sample[3]  # 第4通道灰度图
            gray_image2 = sample[6]  # 第7通道灰度图
            gray_image1 = (gray_image1 * 255).astype(np.uint8)
            gray_image2 = (gray_image2 * 255).astype(np.uint8)

            # 在RGB图像上绘制红绿点
            rgb_image_with_points = rgb_image.copy()
            rgb_image_with_points[red_points == 1] = [255, 0, 0]  # 红色点
            rgb_image_with_points[green_points == 1] = [0, 255, 0]  # 绿色点

            # 显示 RGB 图像
            axs[idx, 0].imshow(rgb_image_with_points)
            axs[idx, 0].set_title(f'RGB Image with Points {idx + 1}')
            axs[idx, 0].axis('off')

            # 在第4通道灰度图上绘制红绿点
            gray_image1_with_points = gray_image1.copy()
            # 将红绿点覆盖到灰度图像上，保持灰度图像的灰度值，只有在对应点位置上绘制彩色点
            gray_image1_with_points = np.stack([gray_image1_with_points] * 3, axis=-1)  # 扩展到3个通道
            gray_image1_with_points[red_points == 1] = [255, 0, 0]  # 红色点
            gray_image1_with_points[green_points == 1] = [0, 255, 0]  # 绿色点

            axs[idx, 1].imshow(gray_image1_with_points)
            axs[idx, 1].set_title(f'Gray Image 4 with Points {idx + 1}')
            axs[idx, 1].axis('off')

            # 在第7通道灰度图上绘制红绿点
            gray_image2_with_points = gray_image2.copy()
            # 将红绿点覆盖到灰度图像上，保持灰度图像的灰度值，只有在对应点位置上绘制彩色点
            gray_image2_with_points = np.stack([gray_image2_with_points] * 3, axis=-1)  # 扩展到3个通道
            gray_image2_with_points[red_points == 1] = [255, 0, 0]  # 红色点
            gray_image2_with_points[green_points == 1] = [0, 255, 0]  # 绿色点

            axs[idx, 2].imshow(gray_image2_with_points)
            axs[idx, 2].set_title(f'Gray Image 7 with Points {idx + 1}')
            axs[idx, 2].axis('off')

            if additional_image is not None:
                if isinstance(additional_image, torch.Tensor):
                    additional_image = additional_image.cpu().numpy()
                add_image = additional_image[idx]
                add_image = np.transpose(add_image, (1, 2, 0))
                add_image = (add_image * 255).astype(np.uint8)
                axs[idx, 3].imshow(add_image)
                axs[idx, 3].set_title(f'Additional Image {idx + 1}')
                axs[idx, 3].axis('off')

        plt.tight_layout()
        # plt.show()
        # print( )
