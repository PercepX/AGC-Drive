import open3d as o3d
import numpy as np
import os

def pcd_to_npy_file(file_path):
    # 使用open3d读取PCD文件
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"Error reading point cloud file: {e}")
        return

    # 获取点云数据，返回的是 Nx3 的numpy数组
    points = np.asarray(pcd.points)
    
    # 构造新的文件路径，替换后缀为.npy
    new_file_path = file_path.replace(".pcd", ".npy")
    
    # 保存为npy文件
    try:
        np.save(new_file_path, points)
        print(f"Point cloud data saved to {new_file_path}")
    except Exception as e:
        print(f"Error saving point cloud data: {e}")

def pcd_to_npy(file_path):
    # 使用open3d读取PCD文件
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"Error reading point cloud file: {e}")
        return

    # 获取点云数据，返回的是 Nx3 的numpy数组
    points = np.asarray(pcd.points)
    
    return points

if __name__ == "__main__":
    # 输入PCD文件路径
    file_path = "/data/datasets/TriCo3D/train/00/2/17396239980.pcd"

    # 调用转换函数
    pcd_to_npy_file(file_path)
