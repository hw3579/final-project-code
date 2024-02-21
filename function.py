import cv2
from matplotlib import pyplot as plt
import numpy as np

def extract_yellow_area(image, distance=20):
    """
    提取图像中的黄色区域。

    参数：
    - image：输入图像
    - distance：扩展轮廓区域的距离，默认为10

    返回值：
    - result：提取后的图像，只保留了扩展后的黄色区域，其他区域设置为白色
    """

    # 将图像转换为RGB格式（OpenCV默认使用BGR格式）在cv库操作时，需要转换为BGR格式
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
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

