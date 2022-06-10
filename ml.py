import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf



model1 = tf.keras.models.load_model('./myModel')
path = 'predict\HORSE-1.jpg'
img = image.load_img(path,target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

classes = model1.predict(x)
print(classes[0,0])
if classes[0,0] < 0.5:
    print('It is a horse')
else:
    print('It is human')