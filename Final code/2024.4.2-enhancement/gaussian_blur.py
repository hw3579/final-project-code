import cv2

# Load the image
image = cv2.imread('./sample9.5.jpg', cv2.IMREAD_GRAYSCALE)

ksize = [21, 41, 61, 81]
for k in ksize:
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

    # Load the original image
    original_image = cv2.imread('./sample9.5.jpg')

    # Create a mask by thresholding the gradient magnitude
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    from matplotlib import pyplot as plt
    plt.imshow(masked_image, cmap='gray')
    plt.show()
    # Display the result
    cv2.imwrite(f'sobel{k}.jpg', brightened_image)
    cv2.imwrite(f'masked{k}.jpg', masked_image)