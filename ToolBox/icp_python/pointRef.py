'''
作者: Houyh
功能: icp点云匹配
'''
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
import os
from scripts.pcd2npy import pcd_to_npy
from scripts.gt_car import CarB2CarA
from scripts.gt_uav import uav2car

from icp import *

def pyicp(src, ref, point_step = 10, gt = None):
    '''
    para:
        src: 源点云路径, npz文件
        ref: 目标点云路径, npz文件
        point_step: 选取点云数量步长, 默认为10
        gt: 初始转换矩阵路径, npz文件, 默认为None, 即没有初始矩阵-->随机
    '''
    # 加载点云数据
    # X = np.load(ref)  # 目标点云
    X = pcd_to_npy(ref)
    X = X[::point_step].T  # 转置为 (3, N) 形式
    # X = X.T
    N = X.shape[1]

    # P = np.load(src)  # 源点云
    P = pcd_to_npy(src)
    P = P[::point_step].T  # 转置为 (3, N) 形式
    # P = P.T
    
    # 如果没有初始转换矩阵，则随机生成
    if gt is not None:
        # 从 gt.npz 中加载变换矩阵
        gt_data = gt

        # 提取旋转矩阵和位移向量
        R = gt_data[:3, :3]  # 旋转部分 3x3
        t = gt_data[:3, 3].reshape(3, 1)  # 平移部分 3x1，reshape 为列向量
    else:
        R, t = randRt()

    # 应用逆变换
    P = ApplyInvTransformation(P, R, t)

    # ICP 算法
    Rr, Tr, num_iter = IterativeClosestPoint(source_pts=P, target_pts=X, tau=10e-6)
    
    # 变换后的点云
    Np = ApplyTransformation(P, Rr, Tr)

    # 估计R、T，变换后的点云、源点云和目标点云
    return Rr, Tr, Np, P, X, R, t

# 随机旋转和平移
def randRt():
    t = np.random.rand(3, 1) * 25.0

    theta = np.random.rand() * 20
    phi = np.random.rand() * 20
    psi = np.random.rand() * 20

    R = Rot.from_euler('zyx', [theta, phi, psi], degrees=True)
    R = R.as_matrix()
    return R, t

# 转换 numpy 数组为 Open3D 点云对象
def numpy_to_o3d_point_cloud(np_points):
    points = np_points.T  # Open3D expects each row to be a point
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 设置点云数据
    return pcd

# Open3D 可视化
def vis_result(X, Np, P, if_p):
    # 创建目标点云、源点云和变换后的源点云
    pcd_X = numpy_to_o3d_point_cloud(X)  # 目标点云
    pcd_P = numpy_to_o3d_point_cloud(P)  # 源点云
    pcd_Np = numpy_to_o3d_point_cloud(Np)  # 变换后的源点云

    # 设置颜色
    pcd_X.paint_uniform_color([0, 0, 1])  # 蓝色
    pcd_P.paint_uniform_color([1, 0, 0])  # 红色
    pcd_Np.paint_uniform_color([0, 1, 0])  # 绿色

    # 保存点云为 PLY 文件
    # o3d.io.write_point_cloud("point_cloud_registration_result.ply", pcd_X)  # 保存目标点云
    # o3d.io.write_point_cloud("point_cloud_registration_result_src.ply", pcd_P)  # 保存源点云
    # o3d.io.write_point_cloud("point_cloud_registration_result_transformed.ply", pcd_Np)  # 保存变换后的点云

    # 设置可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云到可视化器
    vis.add_geometry(pcd_X)
    vis.add_geometry(pcd_Np)
    if if_p:
        vis.add_geometry(pcd_P)

    # 设置点的大小
    opt = vis.get_render_option()
    opt.point_size = 2.0  # 默认值为 5，减小这个值来减少点的大小

    # 可视化
    vis.run()
    vis.destroy_window()

def vis_save_result(X, Np, P, if_p, save_p):
    # 创建目标点云、源点云和变换后的源点云
    pcd_X = numpy_to_o3d_point_cloud(X)  # 目标点云
    pcd_P = numpy_to_o3d_point_cloud(P)  # 源点云
    pcd_Np = numpy_to_o3d_point_cloud(Np)  # 变换后的源点云

    # 设置颜色
    pcd_X.paint_uniform_color([0, 0, 1])  # 蓝色
    pcd_P.paint_uniform_color([1, 0, 0])  # 红色
    pcd_Np.paint_uniform_color([0, 1, 0])  # 绿色

    # 合并点云：目标点云 + 变换后的源点云 + 原始源点云（根据if_p）
    combined_pcd = pcd_X + pcd_Np
    if if_p:
        combined_pcd += pcd_P

    # 保存合并后的点云为PCD文件
    o3d.io.write_point_cloud(save_p, combined_pcd)

    # 设置可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云到可视化器
    vis.add_geometry(pcd_X)
    vis.add_geometry(pcd_Np)
    if if_p:
        vis.add_geometry(pcd_P)

    # 设置点的大小
    opt = vis.get_render_option()
    opt.point_size = 2.0

    # 可视化
    vis.run()
    vis.destroy_window()

def save_icp_transform_to_npz(R, T, input_file):
    """
    将 ICP 计算得到的 R 和 T 组合为 4x4 变换矩阵，并保存为 .npz 文件。
    :param R: 3x3 旋转矩阵 (numpy array)。
    :param T: 3x1 平移向量 (numpy array)。
    :param output_file: 输出 .npz 文件的路径。
    """
    # 构造 4x4 的变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T.flatten()

    # 生成新文件路径
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_trans{ext}"

    # 保存到 .npy 文件
    np.save(output_file, transform_matrix)
    print(f"Transformation matrix saved to {output_file}")

if __name__ == "__main__":
    # 导入
    # src = "datasets/17399341710_front.pcd"
    src = "demo/CA.pcd"
    # ref = "datasets/17399341710_back.pcd"
    ref = "demo/CB.pcd"
    UAV = "datasets/17399341710_uav.pcd"
    gt_uav = uav2car()
    gt_car = CarB2CarA([34.2916658, 108.7765217, 0, 351.529541], [34.2922904, 108.7764162, 0, 351.8041992])
    save_p = "datasets/fusion.pcd"
    # ICP
    _, _, Np, P, X, _, _ = pyicp(src, ref, 1)
    _, _, Uavp, Uav_P, X, _, _ = pyicp(UAV, ref, 1, gt_uav)
    # print("Rotation Estimated : \n{}".format(Rr))
    # print("Translation Estimated : \n{}".format(Tr))
    # # 计算误差
    # Re, te = CalcTransErrors(R, t, Rr, Tr)
    # print("Rotational Error : {}".format(Re))
    # print("Translational Error : {}".format(te))
    # 可视化
    # vis_save_result(X, Np, P, False, save_p)
    vis_result(X, Np, Uavp, False)
    vis_result(X, Np, Uavp, True)
