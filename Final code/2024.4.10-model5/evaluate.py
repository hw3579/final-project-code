import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf


# Load the model
model = tf.keras.models.load_model("model5.h5")
class_names = ['bad', 'good', 'normal', 'other']

batch_size = 1
img_height = int (2880*0.4)
img_width = int (2160*0.4)

folder="./"
# Traverse the files in the "./normal" directory
image_path = []


class_names_match = class_names[3]


folder2 = f"./{class_names_match}"
for filename in os.listdir(folder2):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path.append(os.path.join(folder2, filename))

result = []

# Load the image
for image_path in tqdm(image_path):
    image = cv2.imread(image_path)

    # Preprocess the image
    image = cv2.resize(image, (img_width, img_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    # image = image / 255.0
    image = np.expand_dims(image, axis=0)
   
    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Print the predicted class and confidence
    print("Image:", image_path)
    print("Predicted class:", class_names[predicted_class])
    print("Confidence:", confidence)
    print()  # Add a blank line for better readability

    result.append([image_path, class_names[predicted_class], confidence])

# print(result)
origin_count = len([item[1] for item in result])
print(origin_count)
count = sum(1 for item in result if item[1] == f"{class_names_match}")
print("Number of 'other' in result[1]:", count)
print("Percentage of 'other' in result[1]:", count / origin_count*100, "%")


