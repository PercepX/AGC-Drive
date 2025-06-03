import os

def synchronize_files(base_folder, reference_folder, target_folders):
    """
    根据参考文件夹中的文件名，保留目标文件夹中对应的文件，删除多余文件。
    
    :param base_folder: 数据文件夹的根路径
    :param reference_folder: 参考文件夹的相对路径（如 camera_image_0）
    :param target_folders: 目标文件夹的相对路径列表（如其他文件夹）
    """
    # 获取参考文件夹中的文件名（不包括扩展名）
    ref_path = os.path.join(base_folder, reference_folder)
    reference_files = {os.path.splitext(f)[0] for f in os.listdir(ref_path)}
    
    for folder in target_folders:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue
        
        # 遍历目标文件夹中的文件
        for file in os.listdir(folder_path):
            file_name, ext = os.path.splitext(file)
            if file_name not in reference_files:
                # 删除多余的文件
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# 配置路径
# base_folder = "jihu2.0"
base_folder = "hongqi2.0"
reference_folder = "camera_image_0"
target_folders = ["camera_image_1", "camera_image_2", "camera_image_3", "lidar_point_cloud_0"]

# 执行同步
synchronize_files(base_folder, reference_folder, target_folders)
