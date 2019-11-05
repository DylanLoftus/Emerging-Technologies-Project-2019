from flask import Flask, escape, request, render_template
import keras as kr
from keras.models import load_model

app = Flask(__name__)
model = load_model('saved_model.h5')

# https://flask.palletsprojects.com/en/1.1.x/quickstart/#rendering-templates
@app.route('/')
def canvas():
    return render_template('tempCanvas.html')


if __name__ == "__main__":
    app.run(debug = True)