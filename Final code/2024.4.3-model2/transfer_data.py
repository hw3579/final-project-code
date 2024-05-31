import numpy as np
from tqdm import tqdm
import os
import cv2
import sys
import function

# Define the input and output directories
input_dir = './origin_data/'
output_dir = './fingerprint/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Traverse through the directory structure
for root, dirs, files in os.walk(input_dir):
    for file in tqdm(files):
        if file.endswith('.jpg'):
            # Get the input and output file paths
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))
            
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Read the TIFF image
            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            
            # Process the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = function.enhance_with_LUT(image)
            # image = function.extract_yellow_area(image)
            # image = function.remove_dark_spots(image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = function.fingerprint(image)[1]
            cv2.imwrite(output_path, image)