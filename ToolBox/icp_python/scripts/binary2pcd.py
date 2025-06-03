import numpy as np
import open3d as o3d


def binary_pcd2pcd(file_path):
    """
    将binary编码的pcd文件转换为正常pcd文件
    """
    # 使用open3d读取PCD文件
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"Error reading point cloud file: {e}")
        return
    
    o3d.io.write_point_cloud(file_path, pcd)  # 保存目标点云
    
    print(f"点云已保存为 {file_path}")

def nump2pcd(file_path):
    """
    将npy文件转换为pcd文件
    """
    np_points = np.load(file_path)

    # Open3D expects each row to be a point
    pcd = o3d.geometry.PointCloud()

    # 设置点云数据
    pcd.points = o3d.utility.Vector3dVector(np_points)

    # 构造新的文件路径，替换后缀为.npy
    new_file_path = file_path.replace(".npy", ".pcd")

    # 保存目标点云
    o3d.io.write_point_cloud(new_file_path, pcd)

    print(f"点云已保存为 {new_file_path}")


if __name__ == "__main__":
    file_path_bipcd = "data/dair/000009.pcd"
    file_path_npy = "data/dair/000009.npy"
    # binary_pcd2pcd(file_path_bipcd)
    nump2pcd(file_path_npy)
    