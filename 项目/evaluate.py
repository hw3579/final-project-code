import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf


# Load the image
image_path = "yellow10.520240205_161539_0.jpg"
image = cv2.imread(image_path)


# Load the model
model = tf.keras.models.load_model("yellow.h5")
class_names = ['10', '10.5', '11', '11.5', '12', '8', '8.5', '9', '9.5']

batch_size = 32
img_height = int (540*1.2)
img_width = int (720*1.2)
# Preprocess the image
image = cv2.resize(image, (img_width, img_height))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Make predictions
predictions = model.predict(image)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# Print the predicted class and confidence
print("Predicted class:", class_names[predicted_class])
print("Confidence:", confidence)