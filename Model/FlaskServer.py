from flask import Flask, escape, request, render_template
import keras as kr
import numpy as np
import re
import tensorflow as tf
import base64
from keras.models import load_model
import imageio
import cv2
from PIL import Image
from io import BytesIO


app = Flask(__name__)

# https://flask.palletsprojects.com/en/1.1.x/quickstart/#rendering-templates
# Renders the template 'tempCanvas.html' which is where our web-app is.
@app.route('/')
def canvas():
    return render_template('tempCanvas.html')

# To load the model 
def init():
    model = load_model('saved_model.h5')
    return model

@app.route('/predict' , methods=['POST'])
def predict():
   # https://stackoverflow.com/questions/13279399/how-to-obtain-values-of-request-variables-using-python-and-flask
   # Takes in the data as a base64 image.
   data = request.values['imageBase64']

   # Send that data to the image parser.
   decode = imageParser(data)

   #return responseString
    
def imageParser(data):

   # ref: https://stackoverflow.com/questions/26070547/decoding-base64-from-post-to-use-in-pil
   tmp = re.sub('^data:image/.+;base64,', '', data)
   decode = base64.b64decode(tmp)

   return decode

def predictNumber(file):
    # To be implemented.

# Run the app.
if __name__ == "__main__":
    app.run(debug = True)