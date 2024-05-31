import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf


model = tf.keras.models.load_model("model3d.h5")
batch_size = 1
img_height = int (224)
img_width = int (224)

pic="./normal/20240205_155104.jpg"

img = cv2.imread(pic)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img_width, img_height))
img = np.expand_dims(img, axis=0)

feature_map = model.predict(img)
print(feature_map)