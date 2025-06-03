"""
作者: Houyh
功能: 带可视化日志的批量点云配准与融合
"""
import numpy as np
import open3d as o3d
import os
import csv
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from scripts.pcd2npy import pcd_to_npy
from scripts.gt_car import CarB2CarA
from scripts.gt_uav import uav2car
from icp import *
from scipy.spatial.transform import Rotation as Rot

# 日志颜色配置
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_info(msg):
    print(f"{Colors.OKGREEN}[INFO]{Colors.ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def log_warning(msg):
    print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def log_error(msg):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def log_debug(msg):
    print(f"{Colors.OKBLUE}[DEBUG]{Colors.ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def calc_trans_errors(R_true, t_true, R_est, t_est):
    """计算变换误差"""
    # 旋转误差（角度）
    R_error = R_true @ R_est.T
    theta = np.arccos((np.trace(R_error) - 1) / 2)
    rot_error = np.degrees(np.abs(theta))
    
    # 平移误差（欧氏距离）
    trans_error = np.linalg.norm(t_true - t_est)
    return rot_error, trans_error

def print_transform_stats(name, R, t):
    """打印变换矩阵统计信息"""
    euler_angles = Rot.from_matrix(R).as_euler('zyx', degrees=True)
    log_debug(f"{name}变换参数：")
    log_debug(f"旋转矩阵:\n{R}")
    log_debug(f"欧拉角(zyx): {euler_angles}°")
    log_debug(f"平移向量: {t.flatten()}")

def pyicp(src, ref, point_step=10, gt=None, log_prefix=""):
    '''
    改进版ICP配准函数，增加日志输出
    '''
    try:
        # 加载点云数据
        X = pcd_to_npy(ref)[::point_step].T
        P = pcd_to_npy(src)[::point_step].T

        log_debug(f"{log_prefix}点云尺寸 | 目标: {X.shape[1]} 点 | 源: {P.shape[1]} 点")

        # 初始变换矩阵
        if gt is not None:
            R = gt[:3, :3]
            t = gt[:3, 3].reshape(3, 1)
            log_debug(f"{log_prefix}使用初始变换矩阵：")
            print_transform_stats("初始", R, t)
        # else:
        #     R, t = randRt()
        #     log_warning(f"{log_prefix}未提供初始矩阵，使用随机变换")
        #     print_transform_stats("随机", R, t)

        # 应用逆变换
        P_transformed = ApplyInvTransformation(P, R, t)

        # ICP配准
        start_time = time.time()
        Rr, Tr, num_iter = IterativeClosestPoint(P_transformed, X, tau=10e-6)
        time_cost = time.time() - start_time

        # 计算误差
        rot_error, trans_error = calc_trans_errors(R, t, Rr, Tr)
        
        log_info(f"{log_prefix}ICP完成 | 迭代: {num_iter}次 | 耗时: {time_cost:.2f}s")
        log_info(f"{log_prefix}配准误差 | 旋转: {rot_error:.4f}° | 平移: {trans_error:.4f}m")
        print_transform_stats("估计", Rr, Tr)

        return Rr, Tr, ApplyTransformation(P_transformed, Rr, Tr), P, X, R, t, num_iter

    except Exception as e:
        log_error(f"{log_prefix}ICP处理失败: {str(e)}")
        raise

def process_scenes(dataset_root, scenes):
    """处理场景主函数"""
    total_files = 0
    success_files = 0
    fail_files = 0

    for scene in scenes:
        scene_path = os.path.join(dataset_root, scene)
        if not os.path.isdir(scene_path):
            log_warning(f"跳过无效场景: {scene}")
            continue

        log_info(f"{Colors.HEADER}▶▶ 开始处理场景: {scene}{Colors.ENDC}")
        
        # 获取所有片段
        fragments = [f for f in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, f))]
        log_info(f"发现 {len(fragments)} 个数据片段")

        for fragment in tqdm(fragments, desc=f"处理场景 {scene}", unit="segment"):
            fragment_path = os.path.join(scene_path, fragment)
            log_debug(f"处理片段: {fragment}")

            # 时间戳解析
            t_sec = int(fragment)
            # log_debug(f"{file_prefix} 时间戳: {t_sec:.1f}")

            # 路径验证
            required_dirs = ['front/lidar', 'back/lidar', 'gps/front', 'gps/back']
            if not all([os.path.exists(os.path.join(fragment_path, d)) for d in required_dirs]):
                log_warning(f"片段 {fragment} 缺少必要目录，跳过")
                continue

            # 处理每个点云文件
            front_lidar_path = os.path.join(fragment_path, 'front', 'lidar')
            pcd_files = [f for f in os.listdir(front_lidar_path) if f.endswith('.pcd')]
            
            for pcd_file in tqdm(pcd_files, desc="处理点云", unit="file", leave=False):
                total_files += 1
                file_prefix = f"[{scene}/{fragment}/{pcd_file}]"
                
                try:
                    base_name = os.path.splitext(pcd_file)[0]
                    t_full = float(base_name) / 10
                    # 加载前车点云
                    front_pcd = os.path.join(front_lidar_path, pcd_file)
                    front_cloud = pcd_to_npy(front_pcd)
                    log_debug(f"{file_prefix} 前车点云加载完成，点数: {len(front_cloud)}")

                    # 初始化融合点云
                    fused_cloud = front_cloud
                    
                    # 后车处理流程
                    back_pcd = os.path.join(fragment_path, 'back', 'lidar', pcd_file)
                    if os.path.exists(back_pcd):
                        # 读取GPS数据
                        gps_front_file = os.path.join(fragment_path, 'gps', 'front', f"{t_sec}.csv")
                        gps_back_file = os.path.join(fragment_path, 'gps', 'back', f"{t_sec}.csv")
                        
                        gpsA = read_gps_csv(gps_front_file, t_full)
                        gpsB = read_gps_csv(gps_back_file, t_full)

                        if gpsA and gpsB:
                            log_info(f"{file_prefix} 后车配准开始")
                            RT_init = CarB2CarA(gpsA, gpsB)
                            _, _, back_trans, _, _, _, _, _ = pyicp(
                                back_pcd, front_pcd, 
                                gt=RT_init,
                                log_prefix=f"{file_prefix} [后车]"
                            )
                            fused_cloud = np.vstack((fused_cloud, back_trans.T))
                            log_info(f"{file_prefix} 后车配准完成 | 新增点数: {back_trans.shape[1]}")
                        else:
                            log_warning(f"{file_prefix} 缺少GPS数据，跳过后车配准")

                    # 无人机处理流程
                    top_pcd = os.path.join(fragment_path, 'top', 'lidar', pcd_file)
                    if os.path.exists(top_pcd) and os.path.exists(os.path.join(fragment_path, 'imu', 'top')):
                        imu_file = os.path.join(fragment_path, 'imu', 'top', f"{t_sec}.csv")
                        imu_data = read_imu_csv(imu_file, t_full)
                        
                        if imu_data and gpsA:
                            log_info(f"{file_prefix} 无人机配准开始")
                            RT_init = uav2car(gpsA[3], *imu_data)
                            _, _, uav_trans, _, _, _, _, _ = pyicp(
                                top_pcd, front_pcd,
                                gt=RT_init,
                                log_prefix=f"{file_prefix} [无人机]"
                            )
                            fused_cloud = np.vstack((fused_cloud, uav_trans.T))
                            log_info(f"{file_prefix} 无人机配准完成 | 新增点数: {uav_trans.shape[1]}")
                        else:
                            log_warning(f"{file_prefix} 缺少IMU数据，跳过无人机配准")

                    # 保存结果
                    save_path = save_fused_cloud(fused_cloud, fragment_path, pcd_file)
                    log_info(f"{file_prefix} 融合完成 | 总点数: {len(fused_cloud)} | 保存至: {save_path}")
                    success_files += 1

                except Exception as e:
                    log_error(f"{file_prefix} 处理失败: {str(e)}")
                    fail_files += 1
                    continue

        log_info(f"{Colors.HEADER}◀◀ 场景 {scene} 处理完成{Colors.ENDC}")
        log_info(f"成功率: {success_files/(success_files+fail_files):.1%}")

    # 最终统计
    log_info(f"{Colors.BOLD}=== 处理总结 ===")
    log_info(f"总处理文件: {total_files}")
    log_info(f"成功文件: {success_files} ({success_files/total_files:.1%})")
    log_info(f"失败文件: {fail_files} ({fail_files/total_files:.1%})")

# （以下函数保持原有实现，添加日志调用）
# read_gps_csv、read_imu_csv、save_fused_cloud等函数保持不变，但需在内部添加日志调用

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
    # 命令行参数配置（增加详细说明）
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="点云融合处理器"
    )
    parser.add_argument('--dataset-root', default="/home/beikh/workspace/xtreme1/datasets/datasets", 
                      help='数据集根目录（包含场景文件夹）')
    parser.add_argument('--scenes', nargs='+', default=['all'],
                      help='指定处理场景（空格分隔）或all处理全部')
    parser.add_argument('--point-step', type=int, default=10,
                      help='点云采样步长（1为全采样，数值越大处理越快）')
    
    args = parser.parse_args()

    # 场景选择逻辑
    if 'all' in args.scenes:
        scenes = [d for d in os.listdir(args.dataset_root) 
                if os.path.isdir(os.path.join(args.dataset_root, d))]
        log_info(f"选择处理全部 {len(scenes)} 个场景")
    else:
        scenes = args.scenes
        log_info(f"指定处理 {len(scenes)} 个场景")

    # 启动处理流程
    start_time = time.time()
    try:
        process_scenes(args.dataset_root, scenes)
    except KeyboardInterrupt:
        log_error("用户中断处理！")
    finally:
        log_info(f"总运行时间: {time.time()-start_time:.2f}秒")
