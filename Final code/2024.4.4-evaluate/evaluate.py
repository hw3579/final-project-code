import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf


# Load the model
model = tf.keras.models.load_model("model3d.h5")
class_names = ['horizental_line', 'normal', 'other', 'slope', 'vertical_line']

batch_size = 1
img_height = int (299)
img_width = int (299)

folder = "./test_data"
subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

results = []

for subfolder in subfolders:
    image_paths = []
    for root, dirs, files in os.walk(subfolder):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".tif"):
                image_paths.append(os.path.join(root, filename))

    subfolder_results = []

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)

        image = cv2.resize(image, (img_width, img_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        print("Image:", image_path)
        print("Predicted class:", class_names[predicted_class])
        print("Confidence:", confidence)
        print()

        subfolder_results.append([image_path, class_names[predicted_class], confidence])

    results.append(subfolder_results)

# Calculate hit rate for each subfolder
for i, subfolder_results in enumerate(results):
    total = len(subfolder_results)
    hits = sum(1 for item in subfolder_results if item[1] == 'normal')
    hit_rate = hits / total if total > 0 else 0
    print(f"Hit rate for {subfolders[i]}: {hit_rate*100}%")


