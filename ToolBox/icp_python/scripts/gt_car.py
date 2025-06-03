import numpy as np

def gps_to_enu(lat_A, lon_A, alt_A, lat_B, lon_B, alt_B):
    # WGS84椭球参数
    R_e = 6378137.0  # 赤道半径 (m)
    e = 0.08181919084  # 第一偏心率

    # 计算ENU坐标（简化版，忽略椭球修正）
    d_lon = np.radians(lon_B - lon_A)
    d_lat = np.radians(lat_B - lat_A)
    d_alt = alt_B - alt_A

    E = d_lon * R_e * np.cos(np.radians(lat_A))
    N = d_lat * R_e
    U = d_alt

    return np.array([E, N, U])

def heading_to_rot_matrix(heading):
    theta = np.radians(heading)
    return np.array([
        [np.sin(theta),  np.cos(theta), 0],
        [-np.cos(theta), np.sin(theta), 0],
        [0,             0,            1]
    ])

def CarB2CarA(GPSA, GPSB):
    # 输入数据
    lat_A, lon_A, alt_A = GPSA[0], GPSA[1], GPSA[2]   # 车辆A的GPS
    lat_B, lon_B, alt_B = GPSB[0], GPSB[1], GPSB[2]  # 车辆B的GPS
    heading_A = GPSA[3]  # 车辆A航向角（度）
    heading_B = GPSB[3]  # 车辆B航向角（度）

    # 1. 计算车辆B在车辆A的ENU坐标系中的坐标
    T_B_in_ENU = gps_to_enu(lat_A, lon_A, alt_A, lat_B, lon_B, alt_B)

    # 2. 计算两车的旋转矩阵
    R_A = heading_to_rot_matrix(heading_A)  # ENU -> 车辆A
    R_B = heading_to_rot_matrix(heading_B)  # ENU -> 车辆B

    # 3. 相对旋转矩阵（车辆B -> 车辆A）
    R_B_to_A = R_A.T @ R_B

    # 4. 相对平移向量（车辆B在车辆A坐标系中的位置）
    t_B_to_A = R_A.T @ T_B_in_ENU

    # 5. 构建RT矩阵
    RT = np.eye(4)
    RT[:3, :3] = R_B_to_A
    RT[:3, 3] = t_B_to_A

    return RT