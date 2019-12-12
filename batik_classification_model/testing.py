from keras.models import load_model
from keras.preprocessing import image
from keras import applications
from keras.models import Model,Sequential
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2
import os

img_width, img_height = 128, 128
#loaded_model = load_model('model.h5')
dir_test = './testing/'
col_batik = [i for i in os.listdir(dir_test)]
batik_index = ['Batik Kawung','Batik Megamendung','Batik Nitik','Batik Parang','Batik Sido Luhur','Batik Truntum','Batik Udan Liris','Batik Gedok','Batik Ceplok','Batik Tambal']

#image_path = "mendung.jpg"
#img = image.load_img(image_path, target_size=(img_width, img_height))
#img = np.expand_dims(img, axis=0)

def predict(filename):
	x = load_img(filename, target_size=(img_width,img_height))
	x = img_to_array(x)
	x = np.array(x)*(1/255)
	x = x.reshape((1,) + x.shape)
	result = loaded_model.predict(x)
	answer = np.argmax(result,-1)
	return answer[0]

