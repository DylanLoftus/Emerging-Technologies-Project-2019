#!/usr/bin/env python
# coding: utf-8

# # Training a Model with Keras
# With this model we have to read in bytes from a file. The file type we are working with has the extension .gz and this file type can be unzipped with gzip.


# Import gzip to unzip the file.
import gzip
# Import numpy and call it np.
import numpy as np
# For encoding categorical variables.
import sklearn.preprocessing as pre
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Open the byte image file and put it in a file called file.
with gzip.open('Data/t10k-images-idx3-ubyte.gz', 'rb') as file:
    # Read the contents of the file into a variable named 'file_content'
    file_content = file.read()


# Tells us what type of data is in the file.
type(file_content)


file_content[0:4]


# Read one image from the File

I = file_content[16:800]
type(I)

# Make an array with the content from the file into an array with 28 rows and 28 columns.
image = ~np.array(list(I)).reshape(28,28).astype(np.uint8)

# Plot the image.
plt.imshow(image, cmap="gray")


# # Reading a Label from the Labels File.

# Use gzip to open the labels folder and call it file.
with gzip.open('Data/t10k-labels-idx1-ubyte.gz', 'rb') as file:
    # Read the contents of the file into a variable named labels.
    labels = file.read()


# Change the byte to an int.
int.from_bytes(labels[8:9], byteorder="big")


# Now create a Neural Network.

# We need to import Keras as Keras is used to make our Model.
import keras as kr

# We're going to build a sequential neural network.
model = kr.models.Sequential()

# Add a hidden layer which contains 1000 neurons and set the input layer with 784.
model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
model.add(kr.layers.Dense(units=400, activation='relu'))
# Add a three neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Use gzip to open the training images file.
with gzip.open('Data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

# Use gzip to open the training labels file.
with gzip.open('Data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()

# Convert the data to arrays.
train_img =  (~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0)
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

# Place all the training images into input
inputs = train_img.reshape(60000, 784)

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

print(train_lbl[0], outputs[0])


for i in range(10):
     print(i, encoder.transform([i]))


# Train the model with the inputs and the outputs
# Pass over the dataset ten times 'epochs=10'
model.fit(inputs, outputs, epochs=10, batch_size=100)


with gzip.open('Data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('Data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()
    
test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)


(encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()


# Make output predictions based on the input which in this case is (test_img[4:5])
model.predict(test_img[4:5])

# Show image at index 4 in the array and reshape the plot to a 28 x 28 array
plt.imshow(test_img[4].reshape(28, 28), cmap='gray')

# Save the model so you don't have to train it every time.
model.save('saved_model.h5')




