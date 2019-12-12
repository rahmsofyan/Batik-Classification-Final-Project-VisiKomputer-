from flask import Blueprint,render_template
from flask import request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from werkzeug import secure_filename

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

graph = tf.get_default_graph()
sess = tf.Session()
set_session(sess)

loaded_model = load_model("static/model/VGG16_v1.h5")
batik_name = ['kawung.html', 'megaMendung.html', 'nitikKarawitan.html', 'parang.html', 'sidoLuhur.html',
              'tuntrum.html', 'udanLiris.html', 'gedok.html', 'ceplok.html', 'tambal.html']
batik_index = ['Batik Kawung','Batik Megamendung','Batik Nitik','Batik Parang','Batik Sido Luhur','Batik Truntum','Batik Udan Liris','Batik Gedok','Batik Ceplok','Batik Tambal']

def predict(img):
    img_width, img_height = 128, 128
    x = load_img(img, target_size=(img_width,img_height))
    x = np.array(x)*(1/255)
    x = img_to_array(x)
    x = x.reshape((1,) + x.shape)
    result = []

    global graph
    with graph.as_default():
        set_session(sess)
        result = loaded_model.predict(x)

    predict_list = result.tolist()[0]
    predict_list = dict(zip(batik_index,predict_list))
    predict_list = sorted(predict_list.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
    predict_list = predict_list[:3]
    answer = np.argmax(result,-1)
    return [answer,predict_list]


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
    result = predict(filename_opt)
    #remove(filename)
    ans = result[0][0]
    predict_list = result[1]

    for i in range(len(predict_list)):
        predict_list[i] = list(predict_list[i])
        predict_list[i][1] = str(predict_list[i][1]*100)+'%'
    #return redirect(url_for(bat_name[ans]))
    return render_template(batik_name[ans], filename = filename,predicts=predict_list)

@WebApp.route("/test")
def send():
    return render_template('gedok.html')


a = predict('tes.jpg')
print(a[1])