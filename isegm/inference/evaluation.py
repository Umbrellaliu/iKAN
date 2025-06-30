from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker
import cv2
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks
try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _ = evaluate_sample_base(sample.image, sample.gt_mask, predictor,
                                            sample_id=index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample_base(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, logs_path=None, dataset_name=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, logs_path=None, dataset_name=None,callback=None):
    # 初始化点击器和预测掩码
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    # 创建保存路径
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    sample_path = save_path / f'{sample_id}_3.jpg'

    # 初始化一个列表，用于存储每一步的 image_with_mask
    all_images = []

    with torch.no_grad():
        # 设置输入图像
        predictor.set_input_image(image)

        # 遍历每次点击
        for click_indx in range(max_clicks):
            # 生成下一个点击点
            clicker.make_next_click(pred_mask)

            # 获取更新后的预测概率
            pred_probs = predictor.get_prediction(clicker)

            # 计算预测掩码
            pred_mask = pred_probs > pred_thr

            # 绘制当前步骤的 image_with_mask（仅包含当前点击步骤的点）
            current_clicks = clicker.clicks_list[:click_indx + 1]  # 只取当前点击步骤的点
            current_image_with_mask = draw_with_blend_and_clicks(image, pred_mask,
                                                                 clicks_list=current_clicks)

            # 将当前步骤的图像添加到列表中
            if click_indx in [3]:
                all_images.append(current_image_with_mask)

            # 计算当前 IoU
            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            # 如果达到目标 IoU，提前退出
            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        # 将 gt_mask 转换为 3D 图像并添加到列表中
        gt_mask_3d = np.stack([gt_mask * 255] * 3, axis=-1).astype(np.uint8)  # 转换为 (H, W, 3)
        #image_with_gt_mask = draw_with_blend_and_clicks(image, gt_mask_3d,clicks_list=clicker.clicks_list)

        # 将原始图像和 gt_mask 图像插入到最前面
        all_images.insert(0, image)
        all_images.insert(1, gt_mask_3d)

        # 将所有图像水平拼接成一行
        combined_image = np.concatenate(all_images, axis=1)

        # 保存图像（OpenCV 使用 BGR 格式，因此需要反转通道）
        cv2.imwrite(str(sample_path), combined_image[:, :, ::-1])

    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs