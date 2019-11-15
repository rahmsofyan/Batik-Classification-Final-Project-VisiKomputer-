from flask import Blueprint,render_template,flash,redirect
from vendor import batik_classification as bc

WebApp = Blueprint("controller",__name__)

@WebApp.route("/")
def index():
	return str(bc.predict())

@WebApp.route("/send")
def send():
	return "Send File"

