import cv2
import numpy as np
from database import LUT_control_points, Filepath
import function

# 回调函数，但我们不会在这里改变任何东西
def nothing(x):
    pass

# 读取一张图片
image = cv2.imread(Filepath.file0)  # 请将'your_image_path.jpg'替换为你的图片路径
#image = cv2.resize(image, (720, 540))
cv2.namedWindow('1',cv2.WINDOW_NORMAL)
cv2.resizeWindow('1', 1440, 1080)

# 创建两个滑动条，分别用于调整Canny边缘检测的两个阈值
cv2.createTrackbar('K', '1', 7, 50, nothing)
cv2.createTrackbar('Threshold1', '1', 100, 500, nothing)
cv2.createTrackbar('Threshold2', '1', 200, 500, nothing)

while True:
    # 获取滑动条的当前位置
    k = cv2.getTrackbarPos('K', '1')
    threshold1 = cv2.getTrackbarPos('Threshold1', '1')
    threshold2 = cv2.getTrackbarPos('Threshold2', '1')

    # 确保k是奇数
    k = k if k % 2 != 0 else k+1

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(image, (k, k), 0)

    gray_yellow = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # 使用Canny进行边缘检测
    edges = cv2.Canny(gray_yellow, threshold1, threshold2)
    # 将边缘图像作为蒙版应用在原始图像上
    masked_image = cv2.bitwise_and(image, image, mask=edges)
    
    # 显示蒙版应用后的图像
    cv2.imshow('1', masked_image)
    # 显示模糊后的图片
    #cv2.imshow('1', edges)

    # 等待键盘输入，如果按下'q'键，则退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清除窗口
cv2.destroyAllWindows()