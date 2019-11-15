# Emerging-Technologies-Project-2019

## To Run
* Clone this repository.
* Open cmder or command prompt.
* Cd to the directory "Model".
* Run the command "python FlaskServer.py"
* Visit http://127.0.0.1:5000/ in any browser.

## The main aim of this project is to:
* Create a Keras Model
* Train the model to recognize images using the MNIST dataset
* Use HTML and JavaScript to generate a canvas
* Use a flask server to take in our drawn digit and convert it into a data format our model can understand
* Use our Keras Model to predict what we have drawn on the canvas
* Update HTML with response containing our predicted number

## Research

### Creating a Keras Model
From watching the videos provided on moodle I got a good understanding of how a model is intialized and trained. Reading the [Keras Documentation](https://keras.io/#getting-started-30-seconds-to-keras "Getting Started With Keras") helped a lot too as it explained each part of designing a model.

To start I decided I'd go with a [Sequential Model](https://keras.io/getting-started/sequential-model-guide/).

I added my input layers and my output layer and compiled the model. After that I trained the model with the inputs(images) and outputs(labels). Over 10 epochs it gave an average accuracy result of 94%. I then read in a test image and label from the MNIST dataset and to predict what one of the images in the array was. The argmax of the prediction was was 4 and when the same image was plotted a 4 was drawn.

I then saved this model to be used in the flask server later.

### Creating a canvas in HTML and JavaScript

To create the canvas in HTML I read through the [link](https://www.html5canvastutorials.com/labs/html5-canvas-paint-application/) provided on moodle. I stripped the canvas width and height properties though and changed them to be a height and width of 28 x 28 to match the picture sizes in the MNIST dataset to make it easy on our model. I also added a button which when pressed clears the canvas so you don't have to refresh the page every time you want to make prediciton.

### Sending/Receiving data to and from a flask server
[How to Send a Canvas Image to a Server](https://stackoverflow.com/questions/13198131/how-to-save-an-html5-canvas-as-an-image-on-a-server)

Through research I found that the best way to send a canvas drawing to a server (in this case a flask server) would be to convert it to a DataURL.

After converting the image to a DataURL we then make an AJAX call to our server giving it the method we want to call, the type of call (GET,POST,etc) and the data we want to send which in this case is a base64 image (DataURL). If our call is successful we fill a H3 tag in our canvas with the string created by our Flask Server (number prediction). However if we have been unsuccessful we fill that H3 tag with "Encountered an Error".

### Processing the image in a flask server
[Getting Data from AJAX call](https://stackoverflow.com/questions/13279399/how-to-obtain-values-of-request-variables-using-python-and-flask)

First I took in the data sent from the AJAX call. I then had to get rid of the first part of the string which doesn't represent the image. I then saved that image and used [cv2](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html) to convert the image to grayscale.

After that I used [numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html) to copy the image but flatten it into a one dimensional array as a float. The tyep uint8 didn't seem to work with the model. I then divided that float by 255 to get a 0 or a number greater than 0.

### Predicting the image in the flask server
[Loading an already saved model](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)

First I loaded the model into my flask server. I then called model.predict() on the array I flattened and saved it into a varibale "prediction". I then [converted](https://stackoverflow.com/questions/961632/converting-integer-to-string) the argmax of the "prediction" array to a string and sent that a response to the AJAX call. This response string can be a number between 0 - 9 depending on what has been drawn in the canvas in the webapp.
