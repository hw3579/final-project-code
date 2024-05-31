import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("model1c.h5")
class_names = ['horizental_line', 'normal', 'other', 'slope', 'vertical_line']

batch_size = 1
img_height = int (2880*0.5)
img_width = int (2880*0.5)

folder="./"
# Traverse the files in the "./normal" directory
image_path = []
folder2 = "./other"
for filename in os.listdir(folder2):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path.append(os.path.join(folder2, filename))

# # Create an image dataset from the directory
# image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     directory=folder,
#     labels='inferred',
#     label_mode='int',
#     batch_size=batch_size,
#     image_size=(img_height, img_width),
#     shuffle=False
# )

# # Iterate over the image dataset
# for images, labels in image_dataset:
#     # Preprocess the images
#     # images = images / 255.0

#     # Make predictions
#     predictions = model.predict(images)
#     predicted_classes = np.argmax(predictions, axis=1)
#     confidences = np.max(predictions, axis=1)

#     # Print the predicted classes and confidences
#     for i in range(len(predicted_classes)):
#         from matplotlib import pyplot as plt
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.savefig("../test/" + class_names[predicted_classes[i]] + str(confidences[i]) + ".png")
#         # print("Predicted class:", class_names[predicted_classes[i]])
#         # print("Confidence:", confidences[i])
#         # print()  # Add a blank line for better readability
result = []

# Load the image
for image_path in image_path:
    image = cv2.imread(image_path)

    # Preprocess the image
    image = cv2.resize(image, (img_width, img_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
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

print(result)
print(len([item[1] for item in result]))
count = sum(1 for item in result if item[1] == 'other')
print("Number of 'other' in result[1]:", count)



