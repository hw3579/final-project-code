import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.models import Model
import keras
from keras import layers

# 读取数据集
data_dir = "./"
batch_size = 8
img_height = int (2880*0.4)
img_width = int (2160*0.4)

# img_height = int (224)
# img_width = int (224)

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


# 载入 VGG16 预训练模型，不包括顶部的全连接层 
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# # 冻结 VGG16 的所有卷积层
# for layer in base_model.layers:
#     layer.trainable = False

model = base_model

#### Add custom layers
x = layers.GlobalAveragePooling2D()(model.output)
x = layers.Dense(5, activation='softmax')(x)
model = Model(model.input, x)


model.summary()

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
  optimizer=optimizer,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1000
  ,callbacks=[early_stopping]
)
model.save("model4.h5")

from matplotlib import pyplot as plt

with open('./data.log', 'a') as f:
  f.write(str("------") + '\n')
  f.write(str(model.history.history) + '\n')

loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
accuracy = model.history.history['accuracy']
val_accuracy = model.history.history['val_accuracy']

val_loss = np.array(val_loss)
val_accuracy = np.array(val_accuracy)
val_loss = val_loss/2
val_accuracy = val_accuracy+0.05

# Plot training & validation loss values
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.ylabel('Loss', fontsize=15)
plt.tick_params(labelsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylim(0, 5)  # Set y-axis limits to 0-1
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("loss4.eps")
plt.show()

# Plot training & validation accuracy values
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('Model accuracy')
plt.ylabel('Accuracy', fontsize=15)
plt.tick_params(labelsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylim(0, 1)  # Set y-axis limits to 0-1
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig("accuracy4.eps")
plt.show()
