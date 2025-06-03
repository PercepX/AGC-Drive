import json
import numpy as np

# 地球半径（单位：米）
EARTH_RADIUS = 6378137.0

# 经纬度转换为 ENU 坐标（忽略高度）
def latlon_to_enu(lat, lon, ref_lat, ref_lon):
    x = (lon - ref_lon) * np.cos(np.deg2rad(ref_lat)) * EARTH_RADIUS
    y = (lat - ref_lat) * EARTH_RADIUS
    return np.array([x, y, 0])

# 航向角转换为旋转矩阵
def heading_to_rotation_matrix(heading):
    theta = np.deg2rad(heading)  # 如果是角度，转换为弧度
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R

# 从 JSON 文件读取数据
def read_gps_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 读取两辆车的 GPS 数据
car_a_data = read_gps_data('hongqi2.0/gps_data/1736819636.919386.json')
car_b_data = read_gps_data('jihu2.0/gps_data/1736819636.884967.json')

# 提取每辆车的经纬度和航向角
lat_a, lon_a, heading_a = car_a_data['latitude'], car_a_data['longitude'], car_a_data['heading']
lat_b, lon_b, heading_b = car_b_data['latitude'], car_b_data['longitude'], car_b_data['heading']

# 参考点（车 A 的位置）
ref_lat, ref_lon = lat_a, lon_a

# 计算两车的 ENU 坐标
pos_a = latlon_to_enu(lat_a, lon_a, ref_lat, ref_lon)
pos_b = latlon_to_enu(lat_b, lon_b, ref_lat, ref_lon)

# 平移向量
translation = pos_b - pos_a

# 旋转矩阵
rotation_a = heading_to_rotation_matrix(heading_a)
rotation_b = heading_to_rotation_matrix(heading_b)
relative_rotation = np.dot(rotation_b, np.linalg.inv(rotation_a))  # 相对旋转

# 构造齐次变换矩阵
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = relative_rotation
transformation_matrix[:3, 3] = translation

# 将转换矩阵保存为 .npy 文件
np.save('transformation_matrix.npy', transformation_matrix)

print("Transformation Matrix saved as transformation_matrix.npy")
print(transformation_matrix)
