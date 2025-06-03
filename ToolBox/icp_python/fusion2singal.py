'''
作者: Houyh
功能: 将融合后标注好的obiect框映射到单个点云中
'''
import numpy as np
import json
import open3d as o3d

def ransac_ground_segmentation(point_cloud, distance_threshold=0.2):
    """
    使用 RANSAC 平面拟合分割地面和非地面点云。
    :param point_cloud: Nx3 点云数据。
    :param distance_threshold: 平面距离阈值。
    :return: 非地面点云 (前景)，地面点云。
    """
    # 转换为 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # RANSAC 拟合平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=100)
    ground_points = pcd.select_by_index(inliers)       # 地面点
    foreground_points = pcd.select_by_index(inliers, invert=True)  # 非地面点
    
    return np.asarray(foreground_points.points), np.asarray(ground_points.points)


def is_box_in_point_cloud(box, point_cloud, threshold = 50):
    """
    检查标注框在点云中是否存在（根据落在框内的点数是否超过阈值）。
    :param box: 标注框，包含中心点、尺寸和方向等信息。
    :param point_cloud: 点云数据，形状为 Nx3。
    :param threshold: 判断框是否存在的点数阈值。
    :return: 布尔值，表示框是否存在于点云中。
    """
    # 获取标注框的中心点、尺寸和方向
    center = np.array([box['center_x'], box['center_y'], box['center_z']])
    size = box['size']  # 长宽高
    direction = box['direction']  # 旋转矩阵
    
    # 计算标注框的八个角点（考虑方向旋转）
    half_size = np.array(size) / 2
    corners = [
        center + np.dot(direction, [dx, dy, dz])
        for dx in [-half_size[0], half_size[0]]
        for dy in [-half_size[1], half_size[1]]
        for dz in [-half_size[2], half_size[2]]
    ]
    corners = np.array(corners)
    
    # 检查点云中的点是否落入框内部（简单包围盒检查）
    in_box = np.all((point_cloud >= corners.min(axis=0)) & (point_cloud <= corners.max(axis=0)), axis=1)
    points_in_box_count = np.sum(in_box)  # 统计框内的点数
    
    return points_in_box_count > threshold  # 点数是否超过阈值

def transform_box_to_single_cloud(box, transform_matrix, is_ego = False):
    """
    将标注框从融合点云坐标系转换到单车点云坐标系。
    :param box: 标注框信息，包含中心点、方向、尺寸等。
    :param transform_matrix: 配准矩阵的逆矩阵。
    :return: 转换后的标注框。
    """
    # 如果是ego视角点云，则不转换
    if is_ego:
        return box
    # 提取中心点
    center = np.array([box['center_x'], box['center_y'], box['center_z'], 1])
    # 转换中心点
    new_center = np.dot(transform_matrix, center)
    
    # 提取并转换方向
    direction = np.array(box['direction'])  # 假设方向为3x3旋转矩阵
    new_direction = np.dot(transform_matrix[:3, :3].T, direction)
    
    # 返回新的标注框
    return {
        'center_x': new_center[0],
        'center_y': new_center[1],
        'center_z': new_center[2],
        'direction': new_direction,
        'size': box['size'],  # 尺寸保持不变
        'category': box['category'],  # 类别保持不变
    }

def parse_json_file(json_file):
    """
    解析 JSON 文件并提取标注框数据。
    :param json_file: JSON 文件路径。
    :return: 标注框列表和其他信息。
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    objects = data[1]['objects']  # 假设 JSON 数据的根是一个列表，标注框在 'objects' 中
    boxes = []
    for obj in objects:
        contour = obj['contour']
        size = [contour['size3D']['x'], contour['size3D']['y'], contour['size3D']['z']]
        center = [contour['center3D']['x'], contour['center3D']['y'], contour['center3D']['z']]
        rotation_z = contour['rotation3D']['z']  # 假设只使用 Z 轴旋转角度
        direction = np.array([
            [np.cos(rotation_z), -np.sin(rotation_z), 0],
            [np.sin(rotation_z),  np.cos(rotation_z), 0],
            [0,                   0,                  1]
        ])  # 构造旋转矩阵

        boxes.append({
            'id': obj['id'],
            'className': obj.get('className', 'unknown'),
            'center_x': center[0],
            'center_y': center[1],
            'center_z': center[2],
            'size': size,
            'direction': direction,
            'category': obj.get('classId', -1)
        })
    return boxes

def process_boxes_and_point_cloud(boxes, point_cloud, transform_matrix, is_ego = False, points_threshold = 50):
    """
    处理标注框和点云，包括转换和验证。
    :param boxes: 标注框列表。
    :param point_cloud: 点云数据 (Nx3)。
    :param transform_matrix: 点云配准矩阵。
    :return: 转换后的标注框和验证结果。
    """
    transformed_boxes = []
    for box in boxes:
        transformed_box = transform_box_to_single_cloud(box, transform_matrix, is_ego)
        exists = is_box_in_point_cloud(transformed_box, point_cloud, points_threshold)
        transformed_boxes.append({
            'box': transformed_box,
            'exists_in_point_cloud': exists
        })
    return transformed_boxes

def update_json_with_transformed_boxes(original_json, transformed_boxes, output_file):
    """
    根据转换后的标注框数据更新原始 JSON 文件中的相关字段，并仅保留存在的框。
    :param original_json: 原始 JSON 数据。
    :param transformed_boxes: 转换后的标注框列表。
    :param output_file: 更新后的 JSON 文件路径。
    """
    updated_objects = []  # 用于存储更新后的存在的对象

    for obj, transformed_box in zip(original_json[1]['objects'], transformed_boxes):
        if transformed_box['exists_in_point_cloud']:
            box = transformed_box['box']
            # 更新 contour 中的 center3D 和 rotation3D
            obj['contour']['center3D'] = {
                'x': box['center_x'],
                'y': box['center_y'],
                'z': box['center_z']
            }
            # 将方向矩阵转为 Z 轴旋转角度（假设是绕 Z 轴旋转）
            rotation_z = np.arctan2(box['direction'][1, 0], box['direction'][0, 0])
            obj['contour']['rotation3D'] = {
                'x': 0,
                'y': 0,
                'z': rotation_z
            }
            # 更新 size3D 保持原样
            obj['contour']['size3D'] = {
                'x': box['size'][0],
                'y': box['size'][1],
                'z': box['size'][2]
            }
            # 将更新后的对象加入到新的列表中
            updated_objects.append(obj)

    # 更新 JSON 数据，仅保留存在的对象
    original_json[1]['objects'] = updated_objects

    # 将更新后的数据保存到新的 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(original_json, f, indent=4)

if __name__ == "__main__":
    # 原始 JSON 文件路径
    input_json_file = "/data/datasets/TriCo3D/train/00/cooperative/17396239980.json"
    output_json_file = "updated_annotations_threshold_100.json"
    point_cloud_path = "/data/datasets/TriCo3D/train/00/2/17396239980.npy"
    transform_matrix_path = "/home/beikh/workspace/xtreme1/datasets/datasets/00/1739623998/front/back2front/17396239980.npy"
    is_ego = False
    points_threshold = 0
    point_cloud = np.load(point_cloud_path)  
    transform_matrix = np.load(transform_matrix_path)     

    print(transform_matrix)  
    print(np.linalg.inv(transform_matrix))

    # 读取 JSON 文件
    with open(input_json_file, 'r') as f:
        original_json = json.load(f)

    # 解析标注框
    boxes = parse_json_file(input_json_file)

    # 处理标注框和点云
    results = process_boxes_and_point_cloud(boxes, point_cloud, np.linalg.inv(transform_matrix), is_ego, points_threshold)

    # 更新 JSON 数据
    update_json_with_transformed_boxes(original_json, results, output_json_file)

    print(f"Updated JSON file saved to: {output_json_file}")
