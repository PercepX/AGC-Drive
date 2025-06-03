import zipfile
import os
import shutil

def reorganize_zip_win(input_path, output_path=None):
    # 处理Windows路径分隔符问题
    input_path = os.path.abspath(input_path)
    
    # 创建临时文件路径
    temp_dir = os.path.dirname(input_path)
    temp_name = os.path.basename(input_path) + ".tmp"
    temp_path = os.path.join(temp_dir, temp_name)
    
    try:
        with zipfile.ZipFile(input_path, 'r') as original_zip:
            with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
                for entry in original_zip.infolist():
                    # 跳过特殊系统文件和隐藏文件
                    if any(s in entry.filename for s in ('__MACOSX', '.DS_Store')):
                        continue
                    
                    # 强制使用Unix路径分隔符
                    clean_name = entry.filename.replace('\\', '/')
                    
                    # 跳过空目录条目
                    if clean_name.endswith('/') and entry.file_size == 0:
                        continue
                    
                    # 构建新路径
                    new_name = f'a/{clean_name}'
                    
                    # 处理目录条目
                    if clean_name.endswith('/'):
                        new_name = f'a/{clean_name}'
                        new_zip.writestr(new_name, b'')
                        continue
                    
                    # 复制文件内容
                    with original_zip.open(entry) as f:
                        content = f.read()
                    new_zip.writestr(new_name, content)

        # 关闭原始文件句柄后再进行文件操作
        del original_zip
        del new_zip

        # Windows特有的文件替换逻辑
        if output_path is None:
            backup_path = input_path + ".bak"
            
            # 创建备份文件
            if os.path.exists(input_path):
                shutil.copy2(input_path, backup_path)
            
            # 尝试替换文件
            try:
                os.remove(input_path)
            except PermissionError:
                raise RuntimeError("无法替换原文件，请关闭可能占用该文件的程序")
            
            shutil.move(temp_path, input_path)
            os.remove(backup_path)
            print(f"成功更新文件: {input_path}")
        else:
            shutil.move(temp_path, output_path)
            print(f"已创建新文件: {output_path}")

    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"处理失败: {str(e)}")

# 使用示例（直接修改原文件）
reorganize_zip_win(r'F:\dataset\00\1739623998\front\lidar.zip')

# 可选：创建新文件
# reorganize_zip_win(r'C:\path\to\original.zip', r'C:\path\to\modified.zip')