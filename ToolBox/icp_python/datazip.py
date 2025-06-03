import os
import zipfile
import argparse

def pack_folders(dataset_root, target_dirs, output_dir):
    """打包指定目录中的front/lidar_fusion子文件夹"""
    for dir_name in target_dirs:
        zip_path = os.path.join(output_dir, f"{dir_name}.zip")
        top_dir = os.path.join(dataset_root, dir_name)
        
        if not os.path.exists(top_dir):
            print(f"警告：目录 {top_dir} 不存在，已跳过")
            continue

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for segment in os.listdir(top_dir):
                segment_path = os.path.join(top_dir, segment)
                lidar_fusion_dir = os.path.join(segment_path, 'front', 'lidar_fusion')
                
                if os.path.isdir(lidar_fusion_dir):
                    # 遍历lidar_fusion目录下的所有文件
                    for root, _, files in os.walk(lidar_fusion_dir):
                        for file in files:
                            src_path = os.path.join(root, file)
                            # 计算在zip中的路径（相对于lidar_fusion目录）
                            rel_path = os.path.relpath(src_path, lidar_fusion_dir)
                            arc_path = os.path.join(segment, 'lidar_point_cloud_0', rel_path)
                            print(segment, arc_path)
                            zipf.write(src_path, arc_path)
                    print(f"已添加 {segment}/front/lidar_fusion 到 {zip_path}")
                else:
                    print(f"警告：{segment_path} 中没有lidar_fusion文件夹，已跳过")

        print(f"成功创建压缩包：{zip_path}")

def main():
    parser = argparse.ArgumentParser(description="打包指定目录的front/lidar_fusion文件夹")
    parser.add_argument("--dataset_root", default="/home/beikh/workspace/xtreme1/datasets/datasets", help="数据集根目录路径")
    parser.add_argument("--target_dirs", nargs='+', required=True, help="需要打包的目录列表（如 00 01）")
    parser.add_argument("--output_dir", default="/home/beikh/workspace/xtreme1/datasets/", help="压缩包输出目录")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pack_folders(**vars(args))

if __name__ == "__main__":
    main()
    