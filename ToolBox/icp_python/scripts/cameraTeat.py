import numpy as np

# 相机内参
camera_internal = {
    "fx": 933.4667,
    "fy": 934.6754,
    "cx": 896.4692,
    "cy": 507.3557
}

# 相机外参（旋转矩阵R和平移向量t）
camera_external = [
    -0.7209479393140598, -0.04004438206239668, -0.6918312773097581, 0,
    0.6911177056736608, 0.0317737123271339, -0.7220434530617444, 0,
    0.05089583188421044, -0.9986925846676711, 0.004768189029478265, 0,
    0.009297776700688867, 1.6581292167169648, -1.0197515012137728, 1
]

# 将外参分解为旋转矩阵和平移向量
R = np.array(camera_external[:9]).reshape(3, 3)  # 旋转矩阵
t = np.array(camera_external[9:12])  # 平移向量

# 3D框的8个顶点（例如在世界坐标系中定义的8个顶点）
# 假设3D框的中心为 [1, 2, 3]，长宽高分别为 [l, w, h]，顶点坐标如下：
l, w, h = 4.006, 1.918, 1.6815  # 长宽高
center_world = np.array([-6.4875, -0.9884, 0.7742])  # 中心点

# 3D框的8个顶点在世界坐标系中的坐标
vertices_world = np.array([
    [center_world[0] - l/2, center_world[1] - w/2, center_world[2] - h/2],  # 1
    [center_world[0] + l/2, center_world[1] - w/2, center_world[2] - h/2],  # 2
    [center_world[0] + l/2, center_world[1] + w/2, center_world[2] - h/2],  # 3
    [center_world[0] - l/2, center_world[1] + w/2, center_world[2] - h/2],  # 4
    [center_world[0] - l/2, center_world[1] - w/2, center_world[2] + h/2],  # 5
    [center_world[0] + l/2, center_world[1] - w/2, center_world[2] + h/2],  # 6
    [center_world[0] + l/2, center_world[1] + w/2, center_world[2] + h/2],  # 7
    [center_world[0] - l/2, center_world[1] + w/2, center_world[2] + h/2]   # 8
])

# 将3D框的顶点从世界坐标系转换到相机坐标系
vertices_camera = np.dot(R, vertices_world.T) + t[:, np.newaxis]

# 将3D框的顶点从相机坐标系投影到图像坐标系
def project_to_image(vertices_camera, camera_internal):
    fx, fy, cx, cy = camera_internal["fx"], camera_internal["fy"], camera_internal["cx"], camera_internal["cy"]
    u, v = [], []
    for x, y, z in vertices_camera.T:
        u.append(fx * x / z + cx)
        v.append(fy * y / z + cy)
    return np.array(u), np.array(v)

# 映射到图像上的2D坐标
u, v = project_to_image(vertices_camera, camera_internal)

# 得到2D框的左上和右下两个顶点坐标
u_min, u_max = np.min(u), np.max(u)
v_min, v_max = np.min(v), np.max(v)

print(f"2D框左上顶点坐标: ({u_min}, {v_min})")
print(f"2D框右下顶点坐标: ({u_max}, {v_max})")

# # 左下和右上的坐标
# left_bottom = (u_min, v_max)  # 左下角
# right_top = (u_max, v_min)   # 右上角

# print(f"2D框左下顶点坐标: {left_bottom}")
# print(f"2D框右上顶点坐标: {right_top}")
