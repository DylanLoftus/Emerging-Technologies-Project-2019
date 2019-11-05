from flask import Flask, escape, request, render_template
import keras as kr
import numpy as np
import re
import tensorflow as tf
import base64
from keras.models import load_model
import imageio
import skimage

app = Flask(__name__)

# https://flask.palletsprojects.com/en/1.1.x/quickstart/#rendering-templates
@app.route('/')
def canvas():
    return render_template('tempCanvas.html')

# To load the model 
def init():
    model = load_model('saved_model.h5')
    return model

@app.route('/predict' , methods=['POST'])
def predict():
    print("got to predict")
    # Take in the datat and send it to the image parser.
    imageParser(request.get_data())
   
    

    #return response

# Run the app.
if __name__ == "__main__":
    app.run(debug = True)