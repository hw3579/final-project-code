import os
import cv2
import numpy as np
from tqdm import tqdm
# 存储文件夹名的列表
folder_names = []

# 遍历当前目录下的所有文件和文件夹
for root, dirs, files in os.walk('./'):
    # 将文件夹名添加到列表中
    folder_names.extend(dirs)
    break  # 如果您只想遍历当前目录，可以使用break语句来停止继续遍历子目录

# 获取每个文件夹下的.tiff文件
tiff_files = []
for folder_name in folder_names:
    folder_path = os.path.join('./', folder_name)
    for file in os.listdir(folder_path):
        if file.endswith('.tif'):
            tiff_files.append(os.path.join(folder_path, file))

# 现在您可以使用OpenCV来处理这些.tiff文件
# 例如，您可以使用cv2.imread()函数读取图像
# 标准化和归一化图像
            



# total_data = []

# # 读取tiff_files并使用OpenCV将其标准化和归一化图像
# for tiff_file in tqdm(tiff_files):
#     # 读取图像
#     image = cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED)
    
#     # 标准化图像
#     normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
#     # 归一化图像
#     normalized_image = normalized_image.astype(np.uint8)
    
#     # 在这里可以对标准化和归一化后的图像进行进一步处理
#     normalized_image_array = np.array(normalized_image)

#     # 将normalized_image_array添加到总数组中
#     total_data.append(normalized_image_array)
# # 生成一个总的数组，存储所有图像的二进制信息
# # 生成一个总的数组，用于保存每个文件的normalized_image_array


# total_data

   
