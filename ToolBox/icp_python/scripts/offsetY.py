import numpy as np
import json

def apply_vertical_offset(camera_data, offset_y):
    """
    给定相机外参矩阵和垂直方向的偏移量，返回偏移后的外参矩阵。
    
    参数:
    camera_data (dict): 从JSON文件中读取的相机外参数据。
    offset_y (float): 垂直方向上的偏移量（单位：像素）。
    
    返回:
    numpy.ndarray: 修改后的相机外参矩阵（4x4）。
    """
    # 获取相机外参数据
    camera_external = camera_data["camera_external"]
    
    # 将外参数据转换为 4x4 矩阵
    extrinsic_mat = np.array(camera_external).reshape((4, 4))
    
    # 提取当前的 y 轴平移量（矩阵中的第 1 行第 3 列）
    t_y = extrinsic_mat[0, 3]  # 提取 y 轴的平移量
    
    # 修改 y 轴的平移量，添加偏移
    extrinsic_mat[0, 3] = t_y + offset_y

    # 将修改后的矩阵重新保存回 camera_data 字典
    camera_data["camera_external"] = extrinsic_mat.flatten().tolist()  # 转换为列表格式

    return camera_data

# 示例：从 JSON 文件读取相机外参
def load_camera_extrinsic_from_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)  # 这里假设只取第一个相机的外参

# 将修改后的数据保存到新的 JSON 文件
def save_to_new_json(camera_data, new_json_file):
    with open(new_json_file, 'w') as f:
        json.dump([camera_data], f, indent=4)  # 保存为列表格式

# 使用示例
json_file = 'bag2_jihu/camera_config/1737105471_254955.json'  # 你的 JSON 文件路径
offset_y = 50  # 垂直方向上的偏移量，单位：像素
new_json_file = '0.json'  # 修改后的新 JSON 文件路径

# 加载外参数据
camera_data = load_camera_extrinsic_from_json(json_file)

# 应用垂直偏移
modified_extrinsic_mat = apply_vertical_offset(camera_data, offset_y)

# 将修改后的数据保存到新的 JSON 文件
save_to_new_json(modified_extrinsic_mat, new_json_file)

print(f"修改后的相机外参已保存到 {new_json_file}")