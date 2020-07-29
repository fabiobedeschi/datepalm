import math

from keras import Model
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, UpSampling2D, Input

from src.autoencoder.params import EPOCHS
from src.config import INPUT_SHAPE


def compose_model(filters: list, input_shape: tuple = INPUT_SHAPE, padding: str = 'same'):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Encoder
    encoder = input_layer
    for f in filters:
        encoder = (Conv2D(f, (3, 3), padding=padding))(encoder)
        encoder = (LeakyReLU())(encoder)
        encoder = (MaxPooling2D(pool_size=(2, 2)))(encoder)

    # Decoder
    decoder = encoder
    for f in filters[::-1]:
        decoder = (Conv2D(f, (3, 3), padding=padding))(decoder)
        decoder = (LeakyReLU())(decoder)
        decoder = (UpSampling2D(size=(2, 2)))(decoder)

    # Output layer
    output_layer = (Conv2D(3, (3, 3), padding=padding, activation='sigmoid'))(decoder)

    # Create the model
    model = Model(input_layer, output_layer)

    # Print and return model
    model.summary()
    return model


def lr_scheduler(epoch, lr):
    return lr if epoch < (EPOCHS * 0.1) or lr < 1e-09 else lr * math.exp(-0.05)
