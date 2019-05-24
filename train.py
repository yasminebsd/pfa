from PIL import Image
from keras.layers import Dense, Conv2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

from preprocess import prepareData, PATCH_SIZE


def network(layers):
    # Define the model
    model = Sequential()
    model.add(Conv2D(3 * layers, (3, 3), activation='relu', input_shape=(PATCH_SIZE, PATCH_SIZE, layers)))
    model.add(Conv2D(6 * layers, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(6 * layers, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model


def train(x_train, y_train, x_test, y_test, nb_classes, epochs):
    # X_train, y_train, X_test, y_test = prepareData()
    # generate the model
    model = network(x_train.shape[3])
    # Train the model
    model.compile(Adam(lr=.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    return model

