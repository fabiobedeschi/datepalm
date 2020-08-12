import math

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU

from src.config import INPUT_SHAPE
from src.classifier.params import EPOCHS


def compose_model(filters: list, input_shape=INPUT_SHAPE, padding: str = 'same'):
    # Compose model structure
    model = Sequential()

    model.add(Conv2D(filters=filters[0], kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape, padding=padding))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for f in filters[1:]:
        model.add(Conv2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding=padding))
        model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=64))
    model.add(Dropout(0.5))
    model.add(LeakyReLU())

    model.add(Dense(units=32))
    model.add(LeakyReLU())

    model.add(Dense(units=1, activation='sigmoid'))

    # Print and return model
    model.summary()
    return model


def lr_scheduler(epoch, lr):
    return lr if epoch < (EPOCHS * 0.2) or lr < 1e-09 else lr * math.exp(-0.05)

