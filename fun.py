import cv2
from matplotlib import pyplot as plt
import numpy as np

def extract_yellow_area(image, distance=10):
    """
    提取图像中的黄色区域。

    参数：
    - image：输入图像
    - distance：扩展轮廓区域的距离，默认为10

    返回值：
    - result：提取后的图像，只保留了扩展后的黄色区域，其他区域设置为白色
    """

    # 将图像转换为RGB格式（OpenCV默认使用BGR格式）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 定义黄色的RGB范围
    lower_yellow = np.array([140, 140, 0], dtype="uint8")
    upper_yellow = np.array([255, 255, 180], dtype="uint8")
    
    # 创建黄色区域的掩膜
    mask_yellow = cv2.inRange(image_rgb, lower_yellow, upper_yellow)
    
    # 查找黄色区域的轮廓
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建一个空的轮廓掩膜
    mask_contours = np.zeros_like(mask_yellow)
    
    # 在掩膜上绘制轮廓
    cv2.drawContours(mask_contours, contours, -1, (255), thickness=cv2.FILLED)
    
    # 膨胀轮廓区域以包括周围像素
    kernel = np.ones((distance*2+1, distance*2+1), np.uint8)
    mask_dilated = cv2.dilate(mask_contours, kernel, iterations=1)
    
    # 创建一个结果图像，只保留扩展后的黄色区域
    result = np.zeros_like(image)
    result[mask_dilated == 255] = image[mask_dilated == 255]
    
    # 将扩展区域之外的区域设置为黑色
    result[mask_dilated != 255] = [255, 255, 255]
    
    # 将图像转换回BGR格式
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result


class LUT_control_points:

    profile1 = [(0, 0), (64, 14), (207, 173), (255, 255)]
    profile2 = [(0, 0), (104, 0), (191, 101), (255, 255)]