import numpy as np
from tqdm import tqdm
import os
import cv2
import sys
sys.path.append("..")
import function

# Define the input and output directories
input_dir = './'
output_dir = './dataset/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Traverse through the directory structure
for root, dirs, files in os.walk(input_dir):
    for file in tqdm(files):
        if file.endswith('.tif'):
            # Get the input and output file paths
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))
            
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Read the TIFF image
            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = function.enhance_with_LUT(image)
            image = function.extract_yellow_area(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Rotate the image clockwise by 180 degrees
            for angle in range(0, 360, 180):
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)

                # # 横置
                # rotated_image = cv2.flip(rotated_image, 1)

                # Save the flipped image
                output_file = f"{output_path[:-4]}_{angle}.jpg"
                cv2.imwrite(output_file, rotated_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
