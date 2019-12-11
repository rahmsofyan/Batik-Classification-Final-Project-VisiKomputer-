from flask import Blueprint,render_template
from flask import request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from werkzeug import secure_filename

loaded_model = load_model("static/model/fixmodel.h5")
batik_name = ['kawung.html', 'megaMendung.html', 'nitikKarawitan.html', 'parang.html', 'sidoLuhur.html',
              'tuntrum.html', 'udanLiris.html', 'kawung.html', 'ceplok.html', 'tambal.html', 'parang.html']

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


