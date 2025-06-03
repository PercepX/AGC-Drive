import os
import json
import numpy as np
import csv
from tqdm import tqdm
import multiprocessing as mp
from scripts.gt_car import CarB2CarA
from scripts.gt_uav import uav2car

# --------- 公共函数 ---------

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
        return None

    if is_car:
        gps_back_file = os.path.join(fragment_path, 'gps', 'back', f"{t_sec}.csv")
        gpsB = read_gps_csv(gps_back_file, t_int)
        if gpsB is None:
            print(f"后车 GPS 数据缺失 {t_int}")
            return None
        RT_init = compute_init_rt(gpsA, gpsB, is_car=True)
        icp_path = os.path.join(fragment_path, 'front', 'back2front', f"{timestamp}.npy")
        save_final = os.path.join(fragment_path, 'front', 'back2front', f"{timestamp}_final.npy")
    else:
        imu_file = os.path.join(fragment_path, 'imu', 'top', f"{t_sec}.csv")
        imu_data = read_imu_csv(imu_file, t_int)
        if imu_data is None:
            print(f"无人机 IMU 数据缺失 {t_int}")
            return None
        RT_init = compute_init_rt(gpsA, imu=imu_data, is_car=False)
        icp_path = os.path.join(fragment_path, 'front', 'uav2car', f"{timestamp}.npy")
        save_final = os.path.join(fragment_path, 'front', 'uav2car', f"{timestamp}_final.npy")

    if not os.path.exists(icp_path):
        print(f"ICP 结果缺失 {icp_path}")
        return None

    RT_icp = load_icp_matrix(icp_path)

    RT_init_inv = np.eye(4)
    R = RT_init[:3, :3]
    t = RT_init[:3, 3].reshape(3, 1)
    RT_init_inv[:3, :3] = R.T
    RT_init_inv[:3, 3] = (-R.T @ t).flatten()

    RT_total = RT_icp @ RT_init_inv

    save_rt_matrix(RT_total, save_final)

    return RT_total

# --------- 核心场景处理 ---------

def process_scene(original_dataset_path, new_dataset_path, scene_id):
    original_scene_dir = os.path.join(original_dataset_path, scene_id)
    new_cooperative_dir = os.path.join(new_dataset_path, scene_id, 'cooperative')
    if not os.path.exists(new_cooperative_dir):
        print(f"[{scene_id}] cooperative 不存在，跳过")
        return

    fragment_dirs = [d for d in os.listdir(original_scene_dir)
                     if os.path.isdir(os.path.join(original_scene_dir, d)) and d != 'cooperative']

    for fragment in fragment_dirs:
        front_dir = os.path.join(original_scene_dir, fragment, 'front')

        back2front_dir = os.path.join(front_dir, 'back2front')
        uav2car_dir = os.path.join(front_dir, 'uav2car')

        # back2front
        if os.path.exists(back2front_dir):
            for npy_file in os.listdir(back2front_dir):
                if not npy_file.endswith('.npy') or npy_file.endswith('_final.npy'):
                    continue
                timestamp = os.path.splitext(npy_file)[0]
                final_npy_path = os.path.join(back2front_dir, f"{timestamp}_final.npy")
                json_path = os.path.join(new_cooperative_dir, f"{timestamp}.json")
                if not os.path.exists(json_path):
                    continue

                if os.path.exists(final_npy_path):
                    RT_total = np.load(final_npy_path)
                else:
                    RT_total = compute_final_rt(os.path.join(original_scene_dir, fragment), timestamp, is_car=True)
                    if RT_total is None:
                        continue

                with open(json_path, 'r') as f:
                    data = json.load(f)

                data = [item for item in data if not (isinstance(item, dict) and 'pairwise_t_matrix1' in item)]
                data.insert(0, {'pairwise_t_matrix1': RT_total.tolist()})

                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=4)

        # uav2car
        if os.path.exists(uav2car_dir):
            for npy_file in os.listdir(uav2car_dir):
                if not npy_file.endswith('.npy') or npy_file.endswith('_final.npy'):
                    continue
                timestamp = os.path.splitext(npy_file)[0]
                final_npy_path = os.path.join(uav2car_dir, f"{timestamp}_final.npy")
                json_path = os.path.join(new_cooperative_dir, f"{timestamp}.json")
                if not os.path.exists(json_path):
                    continue

                if os.path.exists(final_npy_path):
                    RT_total = np.load(final_npy_path)
                else:
                    RT_total = compute_final_rt(os.path.join(original_scene_dir, fragment), timestamp, is_car=False)
                    if RT_total is None:
                        continue

                with open(json_path, 'r') as f:
                    data = json.load(f)

                data = [item for item in data if not (isinstance(item, dict) and 'pairwise_t_matrix2' in item)]
                keys = [list(item.keys())[0] for item in data if isinstance(item, dict)]
                insert_index = 1 if 'pairwise_t_matrix1' in keys else 0
                data.insert(insert_index, {'pairwise_t_matrix2': RT_total.tolist()})

                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=4)

# --------- 主入口 ---------

def main():
    original_dataset_path = '/home/beikh/workspace/xtreme1/datasets/datasets'
    new_dataset_path = '/home/beikh/workspace/OpenCOOD-main'

    available_scenes = sorted([f for f in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, f))])

    print("原数据集中的可用场景：", ' '.join(available_scenes))
    selected = input("请输入要处理的场景编号（空格分隔，留空则全部处理）：").strip()

    if selected:
        scene_ids = selected.split()
    else:
        scene_ids = available_scenes

    print(f"将处理以下场景：{scene_ids}")

    with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
        for scene_id in scene_ids:
            pool.apply_async(process_scene, args=(original_dataset_path, new_dataset_path, scene_id))
        pool.close()
        pool.join()

    print("✅ 全部处理完成")

if __name__ == "__main__":
    main()
