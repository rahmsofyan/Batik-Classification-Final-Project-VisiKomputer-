import os
import random
import numpy as np
import json
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import urllib.request


img_width,img_height = 128,128

class BatikClassification():

	def __init__(self,model_path,info_path,encoding="utf-8"):
		self.Model_Classification = load_model(model_path)
		
		with open(info_path,encoding=encoding) as file_json :
			self.Batik_Info = json.load(file_json)

	def predictImage(self,imageBatik=None):
		x = img_to_array(x)
		x = x.reshape((1,) + (img_height,img_width,3))
		result = self.Model_Classification.predict(x).tolist()
		index = np.argmax(result,-1)[0]

		return (index,result[0][index])

	def predictPath(self,imagepath=None):
		x = load_img(imagepath, target_size=(img_width,img_height))
		x = img_to_array(x)
		x = x.reshape((1,) + x.shape)
		result = self.Model_Classification.predict(x).tolist()
		index = np.argmax(result,-1)[0]
		return (index,result[0][index])


	def getInfoBatik(self,index):
		return self.Batik_Info["Batik_Info"][index]
