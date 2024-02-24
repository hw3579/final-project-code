import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
# 存储文件夹名的列表
# folder_names = []

# # 遍历当前目录下的所有文件和文件夹
# for root, dirs, files in os.walk('./dataset/'):
#     # 将文件夹名添加到列表中
#     folder_names.extend(dirs)
#     break  # 如果您只想遍历当前目录，可以使用break语句来停止继续遍历子目录

# # 获取每个文件夹下的.tiff文件
# tiff_files = []
# for folder_name in folder_names:
#     folder_path = os.path.join('./dataset/', folder_name)
#     for file in os.listdir(folder_path):
#         if file.endswith('.jpg'):
#             tiff_files.append(os.path.join(folder_path, file))

# tiff_files
data_dir = "./dataset/"
batch_size = 32
img_height = int (540*1.2)
img_width = int (720*1.2)

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense

# # Load the ResNet50 model with pre-trained weights
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Create a new model on top
# model = Sequential([
#   base_model,
#   Flatten(),
#   Dense(128, activation='relu'),
#   Dense(num_classes, activation='softmax')
# ])

# # Freeze the base model
# base_model.trainable = False


model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100
)

model.save("final.h5")
import matplotlib.pyplot as plt

# Get training history
history = model.history.history

# Plot loss
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





