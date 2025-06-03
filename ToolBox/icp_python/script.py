"""
作者: Houyh
功能: 带可视化日志的批量点云配准与融合（多线程优化版）
"""
import numpy as np
import open3d as o3d
import os
import csv
import argparse
import time
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scripts.pcd2npy import pcd_to_npy
from scripts.gt_car import CarB2CarA
from scripts.gt_uav import uav2car
from icp import *
from scipy.spatial.transform import Rotation as Rot

# 日志颜色配置
class Colors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_info(msg):
    print(f"{Colors.OKGREEN}[INFO]{Colors.ENDC} {datetime.now().strftime('%H:%M:%S')} - {msg}")

def log_error(msg):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {datetime.now().strftime('%H:%M:%S')} - {msg}")

def calc_trans_errors(R_true, t_true, R_est, t_est):
    """计算变换误差"""
    R_error = R_true @ R_est.T
    theta = np.arccos((np.trace(R_error) - 1) / 2)  # 修复括号错误
    rot_error = np.degrees(np.abs(theta))
    trans_error = np.linalg.norm(t_true - t_est)
    return rot_error, trans_error

def pyicp(src, ref, point_step=10, gt=None, log_prefix=""):
    '''
    改进版ICP配准函数，精简日志输出
    '''
    try:
        X = pcd_to_npy(ref)[::point_step].T
        P = pcd_to_npy(src)[::point_step].T

        if gt is not None:
            R = gt[:3, :3]
            t = gt[:3, 3].reshape(3, 1)
        # else:
        #     R, t = randRt()

        P_transformed = ApplyInvTransformation(P, R, t)

        start_time = time.time()
        Rr, Tr, num_iter = IterativeClosestPoint(P_transformed, X, tau=10e-6)
        time_cost = time.time() - start_time

        rot_error, trans_error = calc_trans_errors(R, t, Rr, Tr)
        log_info(f"{log_prefix} ICP完成 | 迭代: {num_iter}次 | 耗时: {time_cost:.2f}s | 旋转误差: {rot_error:.2f}° | 平移误差: {trans_error:.2f}m")

        return Rr, Tr, ApplyTransformation(P_transformed, Rr, Tr), P, X, R, t, num_iter

    except Exception as e:
        log_error(f"{log_prefix} ICP处理失败: {str(e)}")
        raise

def save_rt_matrix(R, t, save_dir, timestamp):
    """保存RT矩阵"""
    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = R
    rt_matrix[:3, 3] = t.flatten()
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{timestamp}.npy"), rt_matrix)

def process_single_file(args):
    """处理单个点云文件"""
    fragment_path, pcd_file, point_step = args
    timestamp = os.path.splitext(pcd_file)[0]
    log_prefix = f"[{os.path.basename(os.path.dirname(fragment_path))}/{pcd_file}]"  # 修复路径处理

    try:
        # 前车处理流程
        front_pcd = os.path.join(fragment_path, 'front', 'lidar', pcd_file)
        fused_cloud = pcd_to_npy(front_pcd)

        # 后车处理
        back_pcd = os.path.join(fragment_path, 'back', 'lidar', pcd_file)
        if os.path.exists(back_pcd):
            t_sec = int(os.path.basename(fragment_path))
            t_full = float(timestamp) / 10
            
            # 替换海象运算符
            gps_front_file = os.path.join(fragment_path, 'gps', 'front', f"{t_sec}.csv")
            gps_back_file = os.path.join(fragment_path, 'gps', 'back', f"{t_sec}.csv")
            
            gpsA = read_gps_csv(gps_front_file, t_full)
            gpsB = read_gps_csv(gps_back_file, t_full)
            if gpsA and gpsB:
                RT_init = CarB2CarA(gpsA, gpsB)
                Rr, Tr, back_trans, *_ = pyicp(
                    back_pcd, front_pcd, 
                    gt=RT_init,
                    point_step=point_step,
                    log_prefix=f"{log_prefix} [后车]"
                )
                save_dir = os.path.join(fragment_path, 'front', 'back2front')
                print(save_dir)
                save_rt_matrix(Rr, Tr, save_dir, timestamp)
                fused_cloud = np.vstack((fused_cloud, back_trans.T))

        # 无人机处理
        top_pcd = os.path.join(fragment_path, 'top', 'lidar', pcd_file)
        if os.path.exists(top_pcd):
            imu_file = os.path.join(fragment_path, 'imu', 'top', f"{t_sec}.csv")
            imu_data = read_imu_csv(imu_file, t_full)
            if imu_data and gpsA:
                RT_init = uav2car(gpsA[3], *imu_data)
                Rr, Tr, uav_trans, *_ = pyicp(
                    top_pcd, front_pcd,
                    gt=RT_init,
                    point_step=point_step,
                    log_prefix=f"{log_prefix} [无人机]"
                )
                save_dir = os.path.join(fragment_path, 'front', 'uav2car')
                save_rt_matrix(Rr, Tr, save_dir, timestamp)
                fused_cloud = np.vstack((fused_cloud, uav_trans.T))

        # 保存融合点云
        save_fused_cloud(fused_cloud, fragment_path, pcd_file)
        return True
    except Exception as e:
        log_error(f"{log_prefix} {str(e)}")
        return False

def process_scenes(dataset_root, scenes, point_step):
    """多线程处理场景主函数"""
    total_files = 0
    success_files = 0
    file_args = []

    # 收集所有待处理文件（增加目录校验）
    for scene in scenes:
        scene_path = os.path.join(dataset_root, scene)
        if not os.path.isdir(scene_path):
            log_error(f"无效场景目录: {scene}")
            continue

        for fragment in os.listdir(scene_path):
            fragment_path = os.path.join(scene_path, fragment)
            if not os.path.isdir(fragment_path):
                continue
            
            front_lidar = os.path.join(fragment_path, 'front', 'lidar')
            if not os.path.exists(front_lidar):
                continue
                
            pcd_files = [f for f in os.listdir(front_lidar) if f.endswith('.pcd')]
            for pcd_file in pcd_files:
                file_args.append((fragment_path, pcd_file, point_step))
                total_files += 1

    # 多线程处理
    with ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:
        futures = [executor.submit(process_single_file, arg) for arg in file_args]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=total_files,
                         desc="处理进度",
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [已用时{elapsed}]"):
            try:
                if future.result():
                    success_files += 1
            except Exception as e:
                log_error(f"处理异常: {str(e)}")

    # 打印统计信息
    log_info(f"{Colors.BOLD}=== 处理总结 ===")
    log_info(f"总处理文件: {total_files}")
    log_info(f"成功文件: {success_files} ({success_files/total_files:.1%})")
    log_info(f"失败文件: {total_files-success_files} ({(total_files-success_files)/total_files:.1%})")

def read_gps_csv(csv_path, target_time):
    """从CSV读取GPS数据"""
    target_time_int = int(round(target_time * 10))
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['Time']) == target_time_int:
                    return [
                        float(row['Latitude']),
                        float(row['Longitude']),
                        0.0,  # 海拔暂设为0
                        float(row['heading'])
                    ]
    except:
        return None

def read_imu_csv(csv_path, target_time):
    """从CSV读取IMU数据"""
    target_time_int = int(round(target_time * 10))
    try:
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
    except:
        return None

def save_fused_cloud(points, fragment_path, filename):
    """保存融合后的点云"""
    # 创建输出目录
    output_dir = os.path.join(fragment_path, 'front', 'lidar_fusion')
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为Open3D格式并保存
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join(output_dir, filename), pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default="/home/beikh/workspace/xtreme1/datasets/datasets", help="数据集根目录路径")
    parser.add_argument('--scenes', nargs='+', default=['all'], help="要处理的场景列表")
    parser.add_argument('--point-step', type=int, default=10, help="点云采样间隔")
    args = parser.parse_args()

    start_time = time.time()
    try:
        scenes = args.scenes if 'all' not in args.scenes else [
            d for d in os.listdir(args.dataset_root) 
            if os.path.isdir(os.path.join(args.dataset_root, d))
        ]
        process_scenes(args.dataset_root, scenes, args.point_step)
    except KeyboardInterrupt:
        log_error("用户中断处理！")
    finally:
        log_info(f"总运行时间: {time.time()-start_time:.2f}秒")