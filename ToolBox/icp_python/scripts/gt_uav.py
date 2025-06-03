# import numpy as np

# # 定义旋转角度（顺时针30°）
# theta_deg = -15
# theta_rad = np.deg2rad(theta_deg)  # 角度转弧度

# # 计算cosθ和sinθ
# cos_theta = np.cos(theta_rad)
# sin_theta = np.sin(theta_rad)

# # 构造绕Z轴顺时针旋转的4×4齐次矩阵
# RT = np.array([
#     [cos_theta,  sin_theta, 0, 0],
#     [-sin_theta, cos_theta, 0, 0],
#     [0,          0,         1, 0],
#     [0,          0,         0, 1]
# ], dtype=np.float32)

# # 保存为.npy文件
# np.save("rotation_z_30_cw.npy", RT)

# # 打印矩阵验证
# print("绕Z轴顺时针旋转30°的4×4矩阵：\n", RT)

import numpy as np

def uav2car(heading, orientation_x, orientation_y, orientation_z, orientation_w):
    # 车辆航向角转旋转矩阵
    theta = np.radians(heading)  # heading转换为弧度
    R_vehicle_ENU = np.array([
        [np.sin(theta), -np.cos(theta), 0],
        [np.cos(theta), np.sin(theta), 0],
        [0, 0, 1]
    ])

    # 无人机四元数转旋转矩阵
    qx, qy, qz, qw = orientation_x, orientation_y, orientation_z, orientation_w
    R_drone_ENU = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

    # 计算相对旋转矩阵
    R_drone_to_vehicle = R_vehicle_ENU.T @ R_drone_ENU

    # RT矩阵（旋转部分）
    RT = np.eye(4)
    RT[:3, :3] = R_drone_to_vehicle

    return RT