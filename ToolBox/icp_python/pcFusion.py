'''
作者: Houyh
功能: 将不同智能体的点云匹配到 Ego 车坐标系并融合成一个点云
'''
from pointRef import *
import numpy as np
import open3d as o3d

def pointFusion(ego, ag_1, ag_2, icp_step = 10, save_path = None, if_vis = False):
    '''
    para:
        ego: ego 车点云路径
        ag_1: ag_1 点云路径
        ag_2: ag_2 点云路径
        save_path: 保存文件路径
        if_vis: 是否可视化融合后点云
    '''
    # icp 点云配准
    _, _, Np_1, _, X, _, _ = pyicp(ag_1, ego, icp_step)
    _, _, Np_2, _, X, _, _ = pyicp(ag_2, ego, icp_step)

    # 配准后点云融合（拼接）
    merged_points = np.hstack((Np_1, Np_2, X))

    # 转换为Open3D点云对象
    pcd_merged_points = numpy_to_o3d_point_cloud(merged_points)

    if if_vis:
         # 设置可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 设置颜色
        pcd_merged_points.paint_uniform_color([0, 0, 1])  # 蓝色

        # 添加点云到可视化器
        vis.add_geometry(pcd_merged_points)

        # 设置点的大小
        opt = vis.get_render_option()
        opt.point_size = 2.0  # 默认值为 5，减小这个值来减少点的大小

        # 可视化
        vis.run()
        vis.destroy_window()

    # 保存为PCD文件
    o3d.io.write_point_cloud(save_path, pcd_merged_points)
    print(f"点云已保存为 {save_path}")

def pointFusion_test(ego, ag_1, icp_step = 10, save_path = None, if_vis = False):
    # icp 点云配准
    R, T, Np_1, _, X, _, _ = pyicp(ag_1, ego, icp_step)
    
    # 保存配准矩阵
    save_icp_transform_to_npz(R, T, ag_1)

    # 配准后点云融合（拼接）
    merged_points = np.hstack((Np_1, X))

    # 转换为Open3D点云对象
    pcd_merged_points = numpy_to_o3d_point_cloud(merged_points)

    if if_vis:
         # 设置可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 设置颜色
        pcd_merged_points.paint_uniform_color([0, 0, 1])  # 蓝色

        # 添加点云到可视化器
        vis.add_geometry(pcd_merged_points)

        # 设置点的大小
        opt = vis.get_render_option()
        opt.point_size = 2.0  # 默认值为 5，减小这个值来减少点的大小

        # 可视化
        vis.run()
        vis.destroy_window()

    # 保存为PCD文件
    o3d.io.write_point_cloud(save_path, pcd_merged_points)
    print(f"点云已保存为 {save_path}")

if __name__ == "__main__":
    ego = "demo/CB.npy"
    ag_1 = "demo/CA.npy"
    ag_2 = "demo/UAV.npy"
    save_path = "demo/fusion_.pcd"
    # pointFusion(ego, ag_1, ag_2, 1, save_path, True)
    pointFusion_test(ego, ag_2, 1, save_path, True)
