import numpy as np

def transform_3d_box(center, size, rotation, R, T):
    """
    将3D框从一个坐标系转换到另一个坐标系
    
    参数:
    center (np.array): 3D框中心点坐标 [x, y, z]
    size (np.array): 3D框的长宽高 [l, w, h]
    rotation (np.array): 旋转矩阵 R (3x3)
    translation (np.array): 平移矩阵 t (3,)
    to_euler (bool): 是否返回欧拉角形式的旋转，默认返回旋转矩阵

    返回:
    np.array: 转换后的3D框中心点坐标 [x', y', z']
    np.array: 转换后的长宽高 [l', w', h']
    np.array: 转换后的旋转矩阵 (或欧拉角)
    """
    # 1. 变换中心点
    transformed_center = np.dot(R, center) + T

    # 2. 变换尺寸（尺寸不变，直接返回）
    transformed_size = size

    # 3. 变换偏转角
    transformed_rotation = np.dot(rotation.T, R)

    return transformed_center, transformed_size, transformed_rotation

# 示例数据
center = np.array([-6.4875, -0.9884, 0.7742])  # 3D框中心点
size = np.array([4.006, 1.918, 1.6815])    # 3D框尺寸 [l, w, h]
rotation = np.array([0, 0, 0.7679])
R = np.array([[ 0.94308079, -0.14206184,  0.30069429],
            [ 0.18370892,  0.97623574, -0.11495566],
            [-0.2772177,   0.1636527,   0.94676721]])
T = np.array([12.08873351, 22.62224161, 2.167372])  # 平移矩阵 T

# 使用转换矩阵映射3D框
new_center, new_size, new_rotation = transform_3d_box(center, size, rotation, R, T)

# 输出结果
print("转换后的中心点坐标:", new_center)
print("转换后的尺寸 (l, w, h):", new_size)
print("转换后的旋转 (欧拉角):", new_rotation.T)
