# CNN

# CNN Model for Image Classification

This repository contains a Convolutional Neural Network (CNN) model implemented in Keras for image classification. The model consists of 5 convolutional layers, each followed by a ReLU activation function and a max-pooling layer. After the convolutional blocks, the network has a dense layer followed by an output layer with 10 neurons, one for each class.

## Description

The goal of this project is to build a flexible CNN model for classifying images from the iNaturalist dataset. The model architecture can be easily customized to change the number of filters, filter sizes, activation functions, and the number of neurons in the dense layer.

## Prerequisites

- Python 3.x
- Keras
- TensorFlow
- NumPy

##  Usage

The following code demonstrates how to build and compile the CNN model:

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten

def build_cnn_model(input_shape, num_filters, filter_size, activation, dense_neurons):
    model = Sequential()

    for i in range(5):
        if i == 0:
            model.add(Conv2D(num_filters, (filter_size, filter_size), input_shape=input_shape))
        else:
            model.add(Conv2D(num_filters, (filter_size, filter_size)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_neurons))
    model.add(Activation(activation))
    model.add(Dense(10))  # Output layer with 10 neurons for 10 classes
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

input_shape = (128, 128, 3)  # Example input shape, modify as per your dataset
num_filters = 16
filter_size = 3
activation = 'relu'
dense_neurons = 128

model = build_cnn_model(input_shape, num_filters, filter_size, activation, dense_neurons)
model.summary()


##  Model Architecture
The CNN model consists of the following layers:

Convolutional Layer
ReLU Activation
Max Pooling
Convolutional Layer
ReLU Activation
Max Pooling
Convolutional Layer
ReLU Activation
Max Pooling
Convolutional Layer
ReLU Activation
Max Pooling
Convolutional Layer
ReLU Activation
Max Pooling
Flatten
Dense Layer
ReLU Activation
Output Layer with 10 neurons (softmax activation)

##  License
This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgements

The project uses the iNaturalist dataset.
Keras and TensorFlow libraries were used for building and training the model.
