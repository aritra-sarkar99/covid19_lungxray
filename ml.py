import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


model = tf.keras.models.load_model("resnet_model.h5")

img_path = "D:\MyProjects\covid19_lungxray\COVID-1050.png"
img = image.load_img(img_path, target_size = (128, 128))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_batch = img_batch * (1./255)
prediction = model.predict(img_batch)[0][0]* 100
print("%.2f" % prediction)
prediction = model.predict(img_batch)[0][1]* 100
print("%.2f" % prediction)
prediction = model.predict(img_batch)[0][2]* 100
print("%.2f" % prediction)

