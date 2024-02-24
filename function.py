import cv2
from matplotlib import pyplot as plt
import numpy as np
from database import LUT_control_points, Filepath

def extract_yellow_area(image, distance=15):
    """
    提取图像中的黄色区域。

    参数：
    - image：输入图像
    - distance：扩展轮廓区域的距离，默认为10

    返回值：
    - result：提取后的图像，只保留了扩展后的黄色区域，其他区域设置为白色
    """


    # 定义黄色的RGB范围
    lower_yellow = np.array([140, 140, 0], dtype="uint8")
    upper_yellow = np.array([255, 255, 180], dtype="uint8")
    
    # 创建黄色区域的掩膜
    mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
    
    # 查找黄色区域的轮廓
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建一个空的轮廓掩膜
    mask_contours = np.zeros_like(mask_yellow)
    
    # 在掩膜上绘制轮廓
    cv2.drawContours(mask_contours, contours, -1, (255), thickness=cv2.FILLED)
    
    distance = int(distance)
    # 膨胀轮廓区域以包括周围像素
    kernel = np.ones((distance*2+1, distance*2+1), np.uint8)
    mask_dilated = cv2.dilate(mask_contours, kernel, iterations=1)
    
    # 将图像转换为RGB格式（OpenCV默认使用BGR格式）在cv库操作时，需要转换为BGR格式
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 创建一个结果图像，只保留扩展后的黄色区域
    result = np.zeros_like(image)
    result[mask_dilated == 255] = image[mask_dilated == 255]
    
    # 将扩展区域之外的区域设置为黑色
    result[mask_dilated != 255] = [255, 255, 255]
    
    # 将图像转换回RGB格式
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result


def canny(image):
    """
    提取图像中的黄色边缘。

    参数：
    image：输入图像，必须是BGR格式的图像。

    返回值：
    edges：提取出的黄色边缘图像。

    """
    #转为BGR格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 高斯滤波
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    # 转为灰度图
    gray_yellow = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # 使用Canny进行边缘检测
    edges = cv2.Canny(gray_yellow, 100, 200)
    #输出RGB格式
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges

def lapliacian(image):

    #转为BGR格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    laplacian = cv2.Laplacian(image,cv2.CV_8U, ksize=3)
    

    # cv2.imshow('laplacian', laplacian)
    # # 将Laplacian转换为8位无符号整数
    # laplacian = cv2.convertScaleAbs(laplacian)
    #输出RGB格式
    laplacian = cv2.cvtColor(laplacian, cv2.COLOR_BGR2RGB)
    return laplacian

def enhance_with_LUT(image):
    
    #RGB输入
    profile = LUT_control_points.profile2

    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Create a full range array
    full_range = np.arange(0, 256)

    # Interpolate the curve using the control points
    curve = np.polyfit([p[0] for p in profile], [p[1] for p in profile], deg=len(profile)-1)
    curve = np.polyval(curve, full_range).astype(np.uint8)

    # Apply the LUT
    enhanced_image = cv2.LUT(image, curve)
    #RGB输出
    return enhanced_image


def fingerprint(image):
    """
    对指纹图像进行处理，包括高斯模糊和二值化处理。

    参数：
    image：输入的指纹图像。

    返回值：
    otsu_threshold：二值化处理的阈值。
    image_binarized：二值化处理后的图像。
    """
    image = cv2.GaussianBlur(image, (5, 5), 0)
    #RGB输入
    otsu_threshold, image_binarized = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #RGB输出
    return otsu_threshold, image_binarized 