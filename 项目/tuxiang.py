import cv2
from matplotlib import pyplot as plt
import numpy as np
image_path = "./../1220240205_163531.jpg"

# img = cv2.imread(image_path)
# img = cv2.resize(img, (1440, 1080))
# cv2.imshow("原始图", img)


# Define control points for the curve
control_points = [(0, 0), (64, 14), (207, 173), (255, 255)]

# Since we don't have the original image path due to the reset, we will assume it is the same
image = cv2.imread(image_path)

# Create a full range array
full_range = np.arange(0, 256)

# Interpolate the curve using the control points
curve = np.polyfit([p[0] for p in control_points], [p[1] for p in control_points], deg=len(control_points)-1)
curve = np.polyval(curve, full_range).astype(np.uint8)

# Apply the LUT
enhanced_image = cv2.LUT(image, curve)


# Display the original and enhanced images
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Enhanced Image')
ax[1].axis('off')

plt.show()

