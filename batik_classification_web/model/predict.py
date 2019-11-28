from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def predict(img):
	img_width, img_height = 128, 128
	x = load_img("mendung.jpg", target_size=(img_width,img_height))
	x = img_to_array(x)
	print(x.shape)
	x = x.reshape((1,) + x.shape)
	print(x.shape)

	loaded_model = load_model("static/model/fixmodel.h5")

	result = loaded_model.predict(x)
	#result = result[0]

	answer = np.argmax(result,-1)
	return result.tolist(), answer
