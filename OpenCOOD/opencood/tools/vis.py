import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def plot_bounding_box_from_corners(corners, ax):
    """
    通过 8 个角点绘制 3D bounding box。

    Parameters
    ----------
    corners : np.array
        8 个角点的坐标，形状为 (8, 3)
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D 坐标轴，用于绘制。
    """
    # 连线 8 个角点，连接成一个立方体的框架
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接底面和顶面
    ]
    
    for edge in edges:
        ax.plot3D(*zip(*corners[edge]), color="r")

# 假设 gt_data 中是 8 个角点坐标 (x, y, z)
# 加载你的点云数据
# 加载点云数据
pcd_data = np.load("opencood/logs/point_pillar_v2vnet_2025_05_12_17_41_06/npy/0000_pcd.npy")  # (N, 3) 或 (N, 4)
gt_data = np.load("opencood/logs/point_pillar_v2vnet_2025_05_12_17_41_06/npy/0000_gt.npy_test.npy")  # (x, y, z, l, w, h, yaw)


# 绘制 3D 可视化图像
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制点云
ax.scatter(pcd_data[:, 0], pcd_data[:, 1], pcd_data[:, 2], s=0.5, label="Point Cloud")

# 对于每个 bounding box，使用 8 个角点绘制立体框
for box in gt_data:
    plot_bounding_box_from_corners(box, ax)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Point Cloud with Bounding Boxes")

# 保存 3D 图像
plt.savefig('point_cloud_with_3d_boxes.png', dpi=300)

# 关闭图形
plt.close()


def plot_bounding_box_2d_from_corners(corners, ax):
    """
    绘制投影到 XY 平面的 bounding box。

    Parameters
    ----------
    corners : np.array
        8 个角点的坐标，形状为 (8, 3)
    ax : matplotlib.axes._subplots.AxesSubplot
        2D 坐标轴，用于绘制。
    """
    # 投影到 XY 平面 (只取前两列)
    corners_2d = corners[:, :2]

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接底面和顶面
    ]
    
    for edge in edges:
        ax.plot(*zip(*corners_2d[edge]), color="r")

# 创建 2D 图像
plt.figure(figsize=(8, 8))
ax = plt.gca()

# 绘制点云（投影到 XY 平面）
plt.scatter(pcd_data[:, 0], pcd_data[:, 1], s=0.5)

# 绘制 bounding boxes 投影到 2D
for box in gt_data:
    plot_bounding_box_2d_from_corners(box, ax)

# 设置坐标轴标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Top-down View (XY plane) with Bounding Boxes')

# 保存 2D 图像
plt.savefig('point_cloud_with_2d_boxes.png', dpi=300)

# 关闭图形
plt.close()
