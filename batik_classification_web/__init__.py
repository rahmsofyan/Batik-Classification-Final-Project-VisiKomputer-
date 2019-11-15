from flask import Flask
print(__name__)

from .controller import WebApp
app = Flask(__name__)

app.register_blueprint(WebApp)
