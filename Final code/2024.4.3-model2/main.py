import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.models import Model
import keras
from keras import layers

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
data_dir = "./fingerprint/"
batch_size = 8
img_height = int (2880*0.5)
img_width = int (2880*0.5)

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

train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))



model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])



#optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)

model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100
  ,callbacks=[early_stopping]
)
model.summary()
model.save("model2c.h5")

from matplotlib import pyplot as plt

with open('./data.log', 'a') as f:
  f.write(str("-------") + '\n')
  f.write(str(model.history.history) + '\n')
# Plot training & validation loss values
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 5)  # Set y-axis limits to 0-1
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("loss2c.eps")
plt.show()

# Plot training & validation accuracy values
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0, 1)  # Set y-axis limits to 0-1
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig("accuracy2c.eps")
plt.show()



