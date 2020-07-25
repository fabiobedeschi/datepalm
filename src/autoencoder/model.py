import math

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, UpSampling2D

from src.config import INPUT_SHAPE, EPOCHS


def compose_model(input_shape=INPUT_SHAPE, padding: str = 'same'):
    # Compose model structure
    model = Sequential()

    # Encoder
    model.add(Conv2D(8, (3, 3), input_shape=input_shape, padding=padding))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, (3, 3), padding=padding))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding=padding))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding=padding))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding=padding))
    model.add(LeakyReLU())

    # Decoder
    model.add(Conv2D(32, (3, 3), padding=padding))
    model.add(LeakyReLU())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding=padding))
    model.add(LeakyReLU())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding=padding))
    model.add(LeakyReLU())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(8, (3, 3), padding=padding))
    model.add(LeakyReLU())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (3, 3), padding=padding, activation='sigmoid'))

    # Print and return model
    model.summary()
    return model


def lr_scheduler(epoch, lr):
    if epoch < (EPOCHS*0.2) or lr < 1e-06:
        return lr
    else:
        return lr * math.exp(-0.05)
