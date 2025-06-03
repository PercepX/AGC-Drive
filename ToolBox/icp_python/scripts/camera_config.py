import os
import json

def expand_camera_config(base_folder, config_folder, reference_folder):
    """
    根据 camera_image_0 中的文件，扩展 camera_config 文件夹中的 0.json 文件。
    :param base_folder: 数据文件夹的根路径
    :param config_folder: 配置文件夹
    :param reference_folder: 参考文件夹
    """
    # 获取参考文件夹中的文件名（不包括扩展名）
    ref_path = os.path.join(base_folder, reference_folder)
    reference_files = sorted(os.listdir(ref_path))  # 按文件名排序，确保顺序一致
    
    # 获取原始的0.json文件内容
    config_file_path = os.path.join(base_folder, config_folder, "0.json")
    if not os.path.exists(config_file_path):
        print(f"{config_file_path} does not exist. Skipping.")
        return
    
    with open(config_file_path, 'r') as f:
        config_data = json.load(f)
    
    # 遍历参考文件夹的文件，生成对应的配置文件
    for i, ref_file in enumerate(reference_files):
        file_name, _ = os.path.splitext(ref_file)
        
        # 复制原始配置数据
        new_config_data = config_data.copy()
        
        # 创建新的 JSON 文件，文件名与参考文件名一致
        new_config_file = os.path.join(base_folder, config_folder, f"{file_name}.json")
        with open(new_config_file, 'w') as f:
            json.dump(new_config_data, f, indent=4)
        
        print(f"Created: {new_config_file}")
        
    # 删除原始的0.json文件
    os.remove(config_file_path)
    print(f"Deleted: {config_file_path}")

# 配置路径
base_folder = "bag2_jihu"
# base_folder = "hongqi2.0"
config_folder = "camera_config"
reference_folder = "camera_image_0"

# 执行扩展
expand_camera_config(base_folder, config_folder, reference_folder)
