import math

from keras import Model
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, UpSampling2D, Input, Dense, Flatten, Reshape, GaussianNoise, \
    Dropout, Activation

from src.autoencoder.params import EPOCHS
from src.config import INPUT_SHAPE, IMG_SIZE


def compose_model(filters: list, input_shape: tuple = INPUT_SHAPE, padding: str = 'same'):
    dim = int(IMG_SIZE / 2 ** len(filters))

    # Input layer
    input_layer = Input(shape=input_shape)

    # Encoder
    encoder = input_layer
    encoder = GaussianNoise(0.01)(encoder)
    for f in filters:
        encoder = (Conv2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding=padding))(encoder)
        encoder = (LeakyReLU())(encoder)
        encoder = (MaxPooling2D(pool_size=(2, 2)))(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(units=512)(encoder)
    encoder = Dropout(0.25)(encoder)
    encoder = LeakyReLU()(encoder)

    # Bottleneck
    encoder = Dense(units=256)(encoder)
    encoder = LeakyReLU()(encoder)

    # Decoder
    decoder = encoder
    decoder = Dense(units=512)(decoder)
    decoder = Dropout(0.25)(decoder)
    decoder = LeakyReLU()(decoder)
    decoder = Dense(units=dim ** 2 * filters[-1])(decoder)
    decoder = LeakyReLU()(decoder)
    decoder = Reshape((dim, dim, -1))(decoder)
    for f in filters[::-1]:
        decoder = (UpSampling2D(size=(2, 2)))(decoder)
        decoder = (Conv2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding=padding))(decoder)
        decoder = (LeakyReLU())(decoder)

    # Output layer
    output_layer = (Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding=padding))(decoder)
    output_layer = Activation(activation='sigmoid')(output_layer)

    # Create the model
    model = Model(input_layer, output_layer)

    # Print and return model
    model.summary()
    return model


def lr_scheduler(epoch, lr):
    return lr if epoch < (EPOCHS * 0.2) or lr < 1e-09 else lr * math.exp(-0.05)
