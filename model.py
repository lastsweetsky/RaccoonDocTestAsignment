import json
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    MaxPooling2D,
    Dropout
)


def get_model():
    """
    Get a convolutional neural network model for angle prediction.

    The model consists of the following layers:

    * A convolutional layer with 50 filters of kernel size 5, with padding='same' and strides=2.
    * A ReLU activation function.
    * A dropout layer with a rate of 0.3.
    * Another convolutional layer with 100 filters of kernel size 3, with padding='same' and strides=1.
    * A max pooling layer with pool_size=(2, 2) and strides=2.
    * Another convolutional layer with 150 filters of kernel size 3, with padding='same' and strides=2.
    * A ReLU activation function.
    * A dropout layer with a rate of 0.3.
    * Another convolutional layer with 200 filters of kernel size 3, with padding='same' and strides=2.
    * A ReLU activation function.
    * A dropout layer with a rate of 0.3.
    * A Flatten layer.
    * A dense layer with 100 neurons, with activation='relu'.
    * A dropout layer with a rate of 0.4.
    * Another dense layer with 200 neurons, with activation='relu'.
    * A dropout layer with a rate of 0.4.
    * A dense layer with 1 neuron, with activation='linear'.

    Returns:
        keras.Model: The convolutional neural network model.
    """

    model = Sequential()

    model.add(
        Conv2D(
            50,
            kernel_size=5,
            padding="same",
            strides=2,
            activation="relu",
            input_shape=(200, 400, 3),
        )
    )
    model.add(Dropout(0.3))

    model.add(Conv2D(100, padding="same", kernel_size=3, strides=1, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(150, kernel_size=3, strides=2, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Conv2D(200, kernel_size=3, strides=2, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Conv2D(200, kernel_size=3, strides=2, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="linear"))

    return model
