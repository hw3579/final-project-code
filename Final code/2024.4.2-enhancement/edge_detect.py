import cv2

# Load the image
image = cv2.imread('./sample9.5.jpg', cv2.IMREAD_GRAYSCALE)

k = 61
low = 5
# apply gussian blur
gaussian_image = cv2.GaussianBlur(image, (k, k), 0)

cv2.imwrite(f'blurred{k}.jpg', gaussian_image)
# Apply the Sobel operator
sobel_x = cv2.Sobel(gaussian_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gaussian_image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude of the gradient
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Normalize the gradient magnitude to the range [0, 255]
gradient_image = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Brighten the gradient image
brightened_image = cv2.add(gradient_image, 50)

# Display the result
# cv2.imwrite(f'sobel{k}.jpg', brightened_image)

# Apply the Laplacian operator
laplacian_image = cv2.Laplacian(gaussian_image, cv2.CV_64F, ksize=5)

# Normalize the Laplacian image to the range [0, 255]
laplacian_image = cv2.normalize(laplacian_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the Laplacian image
# cv2.imwrite(f'laplacian{k}.jpg', laplacian_image)

for low in [2,4,6,8]:
# Apply the Canny edge detector
    canny_image = cv2.Canny(gaussian_image, low, low * 3)

    # Display the Canny image
    cv2.imwrite(f'canny{k}+{low}.jpg', canny_image)

