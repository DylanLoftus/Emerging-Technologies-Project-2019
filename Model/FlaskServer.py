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

   # Open the image convert to bytes.
   img = Image.open(BytesIO(decode))
   # Save the image in bytes.
   img = img.save("image.png")
   # Use openCV to read in the image.
   imgRead = cv2.imread("image.png")
   # Grayscale the image.
   gray = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)

   # Flatten (make one dimensional) and reshape the array without changing it's data.
   # Convert the data to float so we can divide it by 255
   # Dividing by 255 will give us either a 1 or a 0.
   # 1 represents a drawn pixel.
   # 0 represents a pixel that has not been drawn on.
   grayArray2 = np.ndarray.flatten(np.array(gray)).reshape(1, 784).astype("float32") / 255

   print("Printing image to array")
   print(grayArray2)

   #responseString = predictNumber(grayArray2)

   #return responseString

def imageParser(data):
   # ref: https://stackoverflow.com/questions/26070547/decoding-base64-from-post-to-use-in-pil
   tmp = re.sub('^data:image/.+;base64,', '', data)
   decode = base64.b64decode(tmp)

   return decode


# Run the app.
if __name__ == "__main__":
    app.run(debug = True)