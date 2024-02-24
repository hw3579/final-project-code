from database import LUT_control_points, Filepath
import cv2
import bm3d
import numpy as np
import os
import function

# 读取图像
image_path = Filepath.dynamic  # 请替换为你的图像路径
image_path = "yellow1020240124_163438_180.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (1080, 720))
# 产生噪声图像（可选，如果你的图像已经有噪声则不需要）
# noise = np.random.normal(0, 25, image.shape)  # 假设噪声水平为25
# noisy_image = image + noise
# noisy_image = np.clip(noisy_image, 0, 255)  # 限制像素值在0-255

# 应用BM3D算法去噪
# BM3D_result = bm3d.bm3d(image, sigma_psd=0/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
#BM3D_result = cv2.blur(image, (10,10))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.GaussianBlur(image, (13, 13), 0)
image = function.extract_yellow_area(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 保存去噪后的图像
# output_path = 'your_output_path.jpg'  # 请替换为你的输出路径
# cv2.imwrite(output_path, BM3D_result)

# 显示去噪后的图像
cv2.imshow('Denoised Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
