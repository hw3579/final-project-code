import os
import shutil

# 获取./dataset/下的所有子文件夹
subfolders = [f.path for f in os.scandir('./dataset/') if f.is_dir()]

# 遍历每个子文件夹
for folder in subfolders:
    # 获取当前子文件夹下的所有jpg文件
    jpg_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith('.jpg')]
    
    # 遍历每个jpg文件
    for jpg_file in jpg_files:
        # 构建复制后的文件名
        new_file_name = 'ccc' + os.path.basename(jpg_file)
        
        # 构建复制后的文件路径
        new_file_path = os.path.join(folder, new_file_name)
        
        # 复制文件
        shutil.copy(jpg_file, new_file_path)