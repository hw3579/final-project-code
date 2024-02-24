import cv2
from matplotlib import pyplot as plt
import numpy as np

image_path = "./../1220240205_163531.jpg"

img = cv2.imread(image_path)
img = cv2.resize(img, (1440, 1080))
cv2.imshow("原始图", img)

canny_pic = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny_pic = cv2.blur(canny_pic, (3, 3))
edge = cv2.Canny(canny_pic, 3, 9, 3)
# cv2.imshow("边缘提取效果", edge)


gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred_image, 100, 200)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the original image for visualization
image_with_contours = img.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

# Display the images
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.title('Grayscale Image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.title('Canny Edges')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')

#plt.show()


# Adjust contrast
alpha = 1.1  # Contrast control (1.0-3.0)
beta = 0  # Brightness control (0-100)
adjusted_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

adjusted_contrast = cv2.GaussianBlur(adjusted_contrast, (5, 5), 0)
# Convert to HSV to adjust saturation
hsv_image = cv2.cvtColor(adjusted_contrast, cv2.COLOR_BGR2HSV)
saturation_scale = 1.0  # Saturation control (1.0 is no change)
hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], np.array([saturation_scale]))

# Convert back to BGR from HSV
adjusted_saturation = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Display the images
plt.figure(figsize=(36, 12))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(adjusted_contrast, cv2.COLOR_BGR2RGB))
plt.title('Adjusted Contrast')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(adjusted_saturation, cv2.COLOR_BGR2RGB))
plt.title('Adjusted Saturation')


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of yellow color in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Threshold the HSV image to get only yellow colors
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Bitwise-AND mask and original image
yellow_features = cv2.bitwise_and(img, img, mask=mask)

# Convert to grayscale for edge detection
gray = cv2.cvtColor(yellow_features, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny
edges = cv2.Canny(gray, 50, 150)

# Display the results
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.title('edge')
plt.show()