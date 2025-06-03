'''
作者: Houyh
功能: 计算每个框中的点云平均个数
'''
import numpy as np
import json

def load_point_cloud(file_path):
    """
    加载点云数据。
    假设点云文件是 Nx3 的格式，每行代表一个点 (x, y, z)。
    :param file_path: 点云文件路径。
    :return: 点云数据，形状为 Nx3 的 numpy 数组。
    """
    return np.load(file_path)

def calculate_points_in_box(box, point_cloud):
    """
    计算点云中落在标注框内的点的数量。
    :param box: 标注框，包含中心点、尺寸和方向等信息。
    :param point_cloud: 点云数据，形状为 Nx3。
    :return: 落在框内的点数。
    """
    center = np.array([box['center_x'], box['center_y'], box['center_z']])
    size = box['size']  # 长宽高
    direction = box['direction']  # 旋转矩阵

    # 计算标注框的八个角点
    half_size = np.array(size) / 2
    corners = [
        center + np.dot(direction, [dx, dy, dz])
        for dx in [-half_size[0], half_size[0]]
        for dy in [-half_size[1], half_size[1]]
        for dz in [-half_size[2], half_size[2]]
    ]
    corners = np.array(corners)

    # 检查点云中的点是否落入框内部
    in_box = np.all((point_cloud >= corners.min(axis=0)) & (point_cloud <= corners.max(axis=0)), axis=1)
    return np.sum(in_box)  # 返回框内点的数量

def calculate_average_points(json_file, point_cloud_file):
    """
    计算 JSON 文件中所有标注框的平均点云数量。
    :param json_file: JSON 文件路径。
    :param point_cloud_file: 点云文件路径。
    :return: 平均点云数量。
    """
    # 加载 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 加载点云数据
    point_cloud = load_point_cloud(point_cloud_file)

    # 遍历每个标注框并计算点数
    total_points = 0
    num_boxes = 0

    for obj in data[0]['objects']:
        box = {
            'center_x': obj['contour']['center3D']['x'],
            'center_y': obj['contour']['center3D']['y'],
            'center_z': obj['contour']['center3D']['z'],
            'size': [
                obj['contour']['size3D']['x'],
                obj['contour']['size3D']['y'],
                obj['contour']['size3D']['z']
            ],
            'direction': np.eye(3)  # 如果没有旋转矩阵信息，默认单位矩阵
        }
        if 'rotation3D' in obj['contour']:
            rotation_z = obj['contour']['rotation3D']['z']
            box['direction'] = np.array([
                [np.cos(rotation_z), -np.sin(rotation_z), 0],
                [np.sin(rotation_z),  np.cos(rotation_z), 0],
                [0, 0, 1]
            ])

        points_in_box = calculate_points_in_box(box, point_cloud)
        total_points += points_in_box
        num_boxes += 1

    # 计算平均值
    average_points = total_points / num_boxes if num_boxes > 0 else 0
    return average_points

if __name__ == "__main__":
    json_file_path = "data/dair/updated_annotations.json"  # 替换为你的 JSON 文件路径
    point_cloud_file_path = "data/dair/000009_ego.npy"  # 替换为你的点云文件路径

    average_points = calculate_average_points(json_file_path, point_cloud_file_path)
    print(f"每个标注框中的平均点云数量为: {average_points:.2f}")
