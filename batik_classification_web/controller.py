from flask import Blueprint,render_template,flash,redirect
from vendor import batik_classification as bc
from flask import render_template
from flask import request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
from werkzeug import secure_filename
from flask import Flask, redirect, url_for

loaded_model = load_model("static/model/fixmodel.h5")
batik_name = ['kawung.html', 'megaMendung.html', 'nitikKarawitan.html', 'parang.html', 'sidoLuhur.html',
              'tuntrum.html', 'udanLiris.html', 'kawung.html', 'ceplok.html', 'tambal.html', 'parang.html']
"""
bat_name = ['controller.bat0', 
            'controller.bat1',
            'controller.bat2',
            'controller.bat3',
            'controller.bat4',
            'controller.bat5',
            'controller.bat6',
            'controller.bat7',
            'controller.bat8',
            'controller.bat9']
"""

#@WebApp.route('/'+batik_name[0][:-5], methods=['GET'])
def bat0(filename):
    return render_template(batik_name[0])
#@WebApp.route('/'+batik_name[1][:-5], methods=['GET'])
def bat1(filename):
    return render_template(batik_name[1])
#@WebApp.route('/'+batik_name[2][:-5], methods=['GET'])
def bat2(filename):
    return render_template(batik_name[2])
#@WebApp.route('/'+batik_name[3][:-5], methods=['GET'])
def bat3(filename):
    return render_template(batik_name[3])
#@WebApp.route('/'+batik_name[4][:-5], methods=['GET'])
def bat4(filename):
    return render_template(batik_name[4])
#@WebApp.route('/'+batik_name[5][:-5], methods=['GET'])
def bat5(filename):
    return render_template(batik_name[5])
#@WebApp.route('/'+batik_name[6][:-5], methods=['GET'])
def bat6(filename):
    return render_template(batik_name[6])
#@WebApp.route('/'+batik_name[7][:-5], methods=['GET'])
def bat7(filename):
    return render_template(batik_name[7])
#@WebApp.route('/'+batik_name[8][:-5], methods=['GET'])
def bat8(filename):
    return render_template(batik_name[8])
#@WebApp.route('/'+batik_name[9][:-5], methods=['GET'])
def bat9(filename):
    return render_template(batik_name[9])



bat_func =[bat0,
         bat1,
         bat2,
         bat3,
         bat4,
         bat5,
         bat6,
         bat7,
         bat8,
         bat9,
        ]

def predict(img):
    img_width, img_height = 128, 128
    x = load_img(img, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = x.reshape((1,) + x.shape)
    result = loaded_model.predict(x)
    answer = np.argmax(result,-1)
    return answer


WebApp = Blueprint("controller",__name__)

@WebApp.route("/")
def index():
    return render_template('index.html')
    #return str(bc.predict())

@WebApp.route('/upload', methods=['POST'])
def process_file():    
    
    
    images = request.files['image']
    filename = secure_filename(images.filename)
    images.save(os.path.join("static", filename))
    filename_opt = os.path.join("static", filename)
    ans = predict(filename_opt)
    #remove(filename)
    ans = ans[0]
    #return redirect(url_for(bat_name[ans]))
    return render_template(batik_name[ans], filename = filename)

@WebApp.route("/test")
def send():
    return render_template('gedok.html')


