import math
import torch

import math
import torch
import math
import torch
import matplotlib.pyplot as plt

def get_last_point(points):
    # 初始化 last_point 张量
    last_point = torch.zeros((points.shape[0], 1, 4), device=points.device, dtype=points.dtype)

    # 获取最后一个维度的最大值及其索引
    max_values, max_indices = points[:, :, -1].max(dim=1)
    for i in range(points.shape[0]):
        if (points[i, :, -1] == max_values[i]).sum() > 1 or max_values[i] == 100:
            # 重新初始化 last_point
            last_point = torch.full((points.shape[0], 1, 4), -1, device=points.device, dtype=points.dtype)
            return last_point

    # 更新 last_point 的前三个元素
    last_point[:, 0, :3] = points[torch.arange(points.shape[0]).unsqueeze(1), max_indices.unsqueeze(1)].squeeze(1)

    # 根据条件更新 last_point 的最后一个元素
    last_point[:, 0, -1] = (max_indices < points.shape[1] // 2).float()
    #print(last_point)

    return last_point
# def create_positive_mask(center, dist, max_dist):
#     # 使用一个距离衰减函数来创建正向蒙版，基于正向点进行加法增强
#     mask = torch.exp(-dist**2 / (2 * max_dist**2))  # 高斯衰减
#     return mask
# def create_negative_mask(center, dist, max_dist):
#     # 负向蒙版使用反向衰减进行减法
#     mask = 1 - torch.exp(-dist**2 / (2 * max_dist**2))  # 负向衰减
#     return mask



def modulate_prevMask(prev_mask, points, N, R_max):
    with torch.no_grad():
        #R_max=50

        R_min= prev_mask.shape[2]//40

        last_point = get_last_point(points)

        if torch.any(last_point < 0) or last_point.shape[1] > 1:

            return prev_mask

        num_points = points.shape[1] // 2
        row_array = torch.arange(start=0, end=prev_mask.shape[2], step=1, dtype=torch.float64, device=points.device)
        col_array = torch.arange(start=0, end=prev_mask.shape[3], step=1, dtype=torch.float64, device=points.device)
        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)

        prevMod = prev_mask.detach().clone().to(torch.float64)
        prev_mask = prev_mask.detach().clone()

        for bindx in range(points.shape[0]):
            if last_point[bindx, 0, 2] > 100:
                last_point[bindx, 0, 2] = (points[bindx, :, 2] > -1).sum(dim=-1)
            pos_points = points[bindx, :num_points][points[bindx, :num_points, -1] != -1]
            neg_points = points[bindx, num_points:][points[bindx, num_points:, -1] != -1]

            y, x = last_point[bindx, 0, :2]
            p = prev_mask[bindx, 0, y.long(), x.long()]

            dist = torch.sqrt((coord_rows - y) ** 2 + (coord_cols - x) ** 2)
            L2_diff = (prev_mask[bindx, 0] - p) ** 2

            # if last point is positive
            #print(last_point)
            if last_point[bindx, :, -1] == 1:

                # selecting radius
                if neg_points.shape[0] != 0:
                    min_dist = torch.cdist(neg_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)).min(dim=0)[0]
                    r = min_dist / 2
                    r = min(r, R_max)  # 将 r 限制为不大于 R_max
                    modWindow = (dist <= r)
                    if r < R_min:
                        r = R_min
                        modWindow = (dist <= r)
                        if min_dist < R_min:
                            in_modWindow = neg_points[
                                (torch.cdist(neg_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)) < R_min)[:, 0]]
                            for n_click in in_modWindow:
                                dist_n = torch.sqrt((coord_rows - n_click[0]) ** 2 + (coord_cols - n_click[1]) ** 2)
                                modWindow_n = (dist_n <= torch.sqrt((last_point[bindx, 0, 0] - n_click[0]) ** 2 + (
                                            last_point[bindx, 0, 1] - n_click[1]) ** 2))
                                modWindow[modWindow_n] = 0
                                if torch.all(modWindow == 0):
                                    # Set the first position (or any other position) to True to ensure at least one True remains
                                    modWindow[bindx, int(last_point[bindx, 0, 1].item())] = 1  # Ensure indices are integers


                else:
                    r = R_max
                    modWindow = (dist <= r)

                # selecting max gamma
                if p == 0:
                    dist_max = dist[modWindow].max() + 1e-8  # Prevent division by zero
                    prevMod[bindx, 0][modWindow] = 1 - (dist[modWindow] / dist_max)
                    continue
                elif p < 0.99:
                    max_gamma = 1 / (math.log(0.99, p) + 1e-8)  # Prevent log(0) error
                else:
                    max_gamma = 1

                # selecting difference function
                # if last click number is less than N
                #print()
                if last_point[bindx, 0, 2] < N:
                   #  min_value = torch.min(dist)
                   # # if min_value.item() > 4:
                   #  r = torch.tensor(r, dtype=dist.dtype)
                   #  if torch.all(modWindow == 0):
                   #      print(dist[dist <= r])
                   #      print(last_point)
                   #      print(f"dist shape: {dist.shape}")
                   #      # print(f"dist: {dist}")
                   #      # print(f"modWindow: {modWindow}")
                   #      print(f"min(dist): {torch.min(dist).item()}")
                   #      print(f"max(dist): {torch.max(dist).item()}")
                   #
                   #      print(f"The minimum value in dist: {min_value.item()}")
                   #      print(f"Is modWindow all False? {torch.all(modWindow == 0)}")
                   #      print(r)
                   #      print(dist[int(y) - 5:int(y) + 5, int(x) - 5:int(x) + 5])
                   #      print(modWindow.shape)# 打印中心点附近的距离

                    L2_diff_max = L2_diff[modWindow].max() + 1e-8  # Prevent division by zero
                    L2_diff[modWindow] = (L2_diff[modWindow] / L2_diff_max) * 1000
                    diff_th = L2_diff[modWindow].median()
                    exp = -(max_gamma - 1) / (diff_th ** 3 + 1e-8) * (L2_diff[modWindow] - diff_th) ** 3 + 1  # Prevent division by small number
                    exp[exp <= 1] = 1
                else:
                    exp = max_gamma * (1 - (dist[modWindow] / r)) + (dist[modWindow] / r)

                # modulating prev mask
                # pos_mask = create_positive_mask(last_point[bindx, 0, :2], dist, R_max)
                # prevMod[bindx, 0] = prevMod[bindx, 0] + pos_mask  # 加法增强正向区域

                prevMod[bindx, 0][modWindow] = prevMod[bindx, 0][modWindow] ** (1 / (exp + 1e-8))  # Prevent division by zero
                prevMod[bindx, 0][modWindow] = (1 - (dist / r))[modWindow] + prevMod[bindx, 0][modWindow]
                prevMod[bindx, 0][int(y.round()), int(x.round())] = 1


            # if last point is negative
            else:
                # selecting radius
                if pos_points.shape[0] != 0:
                    min_dist = torch.cdist(pos_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)).min(dim=0)[0]
                    r = min_dist / 2
                    r = min(r, R_max)  # 将 r 限制为不大于 R_max
                    modWindow = (dist <= r)
                    if r < R_min:
                        r = R_min
                        modWindow = (dist <= r)
                        if min_dist < R_min:
                            in_modWindow = pos_points[
                                (torch.cdist(pos_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)) < R_min)[:, 0]]
                            for p_click in in_modWindow:
                                dist_p = torch.sqrt((coord_rows - p_click[0]) ** 2 + (coord_cols - p_click[1]) ** 2)
                                modWindow_p = (dist_p <= torch.sqrt((last_point[bindx, 0, 0] - p_click[0]) ** 2 + (
                                            last_point[bindx, 0, 1] - p_click[1]) ** 2))
                                modWindow[modWindow_p] = 0
                                if torch.all(modWindow == 0):
                                    # Set the first position (or any other position) to True to ensure at least one True remains
                                    modWindow[bindx, int(last_point[bindx, 0, 1].item())] = 1  # Ensure indices are integers
                else:
                    r = R_max
                    r = min(r, R_max)  # 将 r 限制为不大于 R_max
                    modWindow = (dist <= r)
                # selecting max gamma
                if p == 1:
                    dist_max = dist[modWindow].max() + 1e-8  # Prevent division by zero
                    prevMod[bindx, 0][modWindow] = dist[modWindow] / dist_max
                    continue
                elif p > 0.01:
                    max_gamma = math.log(0.01, p) + 1e-8  # Prevent log(0) error
                else:
                    max_gamma = 1

                # selecting difference function
                # if last click number is less than N
                if last_point[bindx, 0, 2] < N:
                    L2_diff_max = L2_diff[modWindow].max() + 1e-8  # Prevent division by zero
                    L2_diff[modWindow] = (L2_diff[modWindow] / L2_diff_max) * 1000
                    diff_th = L2_diff[modWindow].median()
                    exp = -(max_gamma - 1) / (diff_th ** 3 + 1e-8) * (L2_diff[modWindow] - diff_th) ** 3 + 1  # Prevent division by small number
                    exp[exp <= 1] = 1
                else:
                    exp = max_gamma * (1 - (dist[modWindow] / r)) + (dist[modWindow] / r)

                # modulating prev mask
                prevMod[bindx, 0][modWindow] = prevMod[bindx, 0][modWindow] ** (exp + 1e-8)  # Prevent division by zero
                prevMod[bindx, 0][modWindow] =  -(1 - (dist / r))[modWindow]+prevMod[bindx, 0][modWindow]
                prevMod[bindx, 0][int(y.round()), int(x.round())] = 0

        # mask = torch.zeros((256, 256), dtype=torch.double,device='cuda')
        # mask[modWindow] = (1 - (dist / r))[modWindow]
        # if last_point[bindx, :, -1] == 1:  # 正向点
        #     #mask = 1 - (dist / r)  # 距离越远蒙版值越暗
        #     prevMod[bindx, 0][modWindow] = mask[modWindow]+prevMod[bindx, 0][modWindow]  #
        # else:  # 负向点
        #     #mask = 1 - (dist / r)  # 距离越远蒙版值越亮
        #     print(mask.shape)
        #     print(prevMod.shape)
        #     print(dist.shape)
        #     print(modWindow.shape)
        #     print(prevMod[bindx, 0].shape)
        #     print(prevMod[bindx, 0][modWindow].shape)
        #
        #     prevMod[bindx, 0][modWindow] = prevMod[bindx, 0][modWindow]-mask[modWindow]
        #
        # #prevMod[bindx, 0][modWindow] = torch.clamp(prevMod[bindx, 0][modWindow], 0, 1)
        # #print(prevMod[bindx, 0][modWindow])#
        #
        # # 限制蒙版范围在[0, 1]之间
        # #mask = torch.clamp(mask, 0, 1)

        #确保调制结果在[0, 1]之间
        prevMod[prevMod>1]=1
        prevMod[prevMod < 0] = 0
        # print(last_point[bindx, :, -1] )
        # print(r)
        #
        # def plot_prevMod(prevMod, bindx):
        #     # 选择当前 batch 的第 bindx 个元素，并去掉维度 1
        #     image = prevMod.cpu() # 如果在 GPU 上需要 .cpu()
        #
        #     # 创建图像
        #     plt.figure(figsize=(6, 6))
        #     plt.imshow(image, cmap='gray')  # 使用灰度色图来显示图像
        #     plt.title(f"prevMod Visualization (bindx = {bindx})")
        #     plt.axis('off')  # 关闭坐标轴
        #     plt.colorbar()  # 显示色条（可选）
        #     plt.show()
        #
        # # 使用示例
        # # 假设 prevMod 是你的 tensor，bindx 是你要查看的 batch index
        # plot_prevMod(prevMod[bindx, 0], bindx=0)
        #
        # print(prevMod.to(torch.float32))


    return prevMod.to(torch.float32)