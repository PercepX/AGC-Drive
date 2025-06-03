import os
import numpy as np
from scripts.gt_car import CarB2CarA
from scripts.gt_uav import uav2car
import csv
from tqdm import tqdm

def load_icp_matrix(path):
    return np.load(path)

def compute_init_rt(gpsA, gpsB=None, imu=None, is_car=True):
    if is_car:
        return CarB2CarA(gpsA, gpsB)
    else:
        return uav2car(gpsA[3], *imu)

def save_rt_matrix(RT, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, RT)

def read_gps_csv(csv_path, target_time_int):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['Time']) == target_time_int:
                return [
                    float(row['Latitude']),
                    float(row['Longitude']),
                    0.0,  
                    float(row['heading'])
                ]
    return None

def read_imu_csv(csv_path, target_time_int):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['timestamp']) == target_time_int:
                return (
                    float(row['orientation_x']),
                    float(row['orientation_y']),
                    float(row['orientation_z']),
                    float(row['orientation_w'])
                )
    return None

def compute_final_rt(fragment_path, timestamp, is_car=True):
    t_sec = int(os.path.basename(fragment_path))
    t_full = float(timestamp) / 10
    t_int = int(round(t_full * 10))

    gps_front_file = os.path.join(fragment_path, 'gps', 'front', f"{t_sec}.csv")
    gpsA = read_gps_csv(gps_front_file, t_int)
    if gpsA is None:
        print(f"前车 GPS 数据缺失 {t_int}")
        return

    if is_car:
        gps_back_file = os.path.join(fragment_path, 'gps', 'back', f"{t_sec}.csv")
        gpsB = read_gps_csv(gps_back_file, t_int)
        if gpsB is None:
            print(f"后车 GPS 数据缺失 {t_int}")
            return
        RT_init = compute_init_rt(gpsA, gpsB, is_car=True)
        icp_path = os.path.join(fragment_path, 'front', 'back2front', f"{timestamp}.npy")
        save_final = os.path.join(fragment_path, 'front', 'back2front', f"{timestamp}_final.npy")
    else:
        imu_file = os.path.join(fragment_path, 'imu', 'top', f"{t_sec}.csv")
        imu_data = read_imu_csv(imu_file, t_int)
        if imu_data is None:
            print(f"无人机 IMU 数据缺失 {t_int}")
            return
        RT_init = compute_init_rt(gpsA, imu=imu_data, is_car=False)
        icp_path = os.path.join(fragment_path, 'front', 'uav2car', f"{timestamp}.npy")
        save_final = os.path.join(fragment_path, 'front', 'uav2car', f"{timestamp}_final.npy")

    if not os.path.exists(icp_path):
        print(f"ICP 结果缺失 {icp_path}")
        return

    RT_icp = load_icp_matrix(icp_path)

    RT_init_inv = np.eye(4)
    R = RT_init[:3, :3]
    t = RT_init[:3, 3].reshape(3, 1)
    RT_init_inv[:3, :3] = R.T
    RT_init_inv[:3, 3] = (-R.T @ t).flatten()

    RT_total = RT_icp @ RT_init_inv

    save_rt_matrix(RT_total, save_final)

def batch_process_all(base_path):
    fragments = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    print(f"共找到 {len(fragments)} 个场景")

    for frag in tqdm(fragments):
        # 后车->前车
        back2front_dir = os.path.join(frag, 'front', 'back2front')
        if os.path.exists(back2front_dir):
            npy_files = [f for f in os.listdir(back2front_dir) if f.endswith('.npy') and not f.endswith('_final.npy')]
            for f in npy_files:
                timestamp = os.path.splitext(f)[0]
                compute_final_rt(frag, timestamp, is_car=True)

        # 无人机->前车
        uav2car_dir = os.path.join(frag, 'front', 'uav2car')
        if os.path.exists(uav2car_dir):
            npy_files = [f for f in os.listdir(uav2car_dir) if f.endswith('.npy') and not f.endswith('_final.npy')]
            for f in npy_files:
                timestamp = os.path.splitext(f)[0]
                compute_final_rt(frag, timestamp, is_car=False)

if __name__ == "__main__":
    base_path = "/home/beikh/workspace/xtreme1/datasets/datasets/00"
    batch_process_all(base_path)
