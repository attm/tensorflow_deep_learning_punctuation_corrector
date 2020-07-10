from flask import Flask, render_template, request
import os
import tensorflow as tf
from main.main_predict import load_models
from encdec_model.predictor import EncDecPredictor


enc, dec, tkn = load_models()
predictor = EncDecPredictor(enc, dec, tkn)

# Flask app setting
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html", output="out text")

@app.route('/', methods=["POST"])
def correct_text():
    txt = request.form.get("input_text")
    return render_template("index.html", output=predictor.predict(txt))

if __name__ == '__main__':
    app.run()