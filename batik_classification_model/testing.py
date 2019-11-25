from keras.models import load_model
from keras.preprocessing import image
from keras import applications
from keras.models import Model,Sequential
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2

img_width, img_height = 128, 128

#image_path = "mendung.jpg"
#img = image.load_img(image_path, target_size=(img_width, img_height))
#img = np.expand_dims(img, axis=0)

x = load_img("mendung.jpg", target_size=(img_width,img_height))
x = img_to_array(x)
print(x.shape)
x = x.reshape((1,) + x.shape)
print(x.shape)

loaded_model = load_model('fixmodel.h5')

result = loaded_model.predict(x)
#result = result[0]

answer = np.argmax(result,-1)
print(result.tolist())
print(answer)
#print(loaded_model.predict_generator(img,1))
