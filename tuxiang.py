import cv2
from matplotlib import pyplot as plt
import numpy as np
from database import LUT_control_points, Filepath
import function


image_path = Filepath.file12
profile = LUT_control_points.profile2



# Since we don't have the original image path due to the reset, we will assume it is the same
image = cv2.imread(image_path)

# 读取为RGB格式

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

##图像为RGB格式

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
#image = cv2.resize(image, (720, 540))
image = cv2.GaussianBlur(image, (5, 5), 0)
# Create a full range array
full_range = np.arange(0, 256)

# Interpolate the curve using the control points
curve = np.polyfit([p[0] for p in profile], [p[1] for p in profile], deg=len(profile)-1)
curve = np.polyval(curve, full_range).astype(np.uint8)

# Apply the LUT
enhanced_image = cv2.LUT(image, curve)

#图像为RGB格式


otsu_threshold, image_binarized = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show the images
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_binarized, cmap='gray')
plt.title(f"Binarized Image with Otsu's threshold: {otsu_threshold}")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(enhanced_image) #RGB格式输出
plt.title('Enhanced Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(function.extract_yellow_area(enhanced_image,10))
plt.title('Yellow Area')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(function.canny(enhanced_image))
plt.title('Enhanced Image')
plt.axis('off')



plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

