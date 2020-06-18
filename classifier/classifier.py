import datetime
import math
import os

import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.utils.vis_utils import plot_model

# Parameters
IMG_SIZE = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
DATA_DIR = '../dataset'
TRAIN_DIR = f'{DATA_DIR}/train'
TEST_DIR = f'{DATA_DIR}/test'
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
STEPS_MULTIPLIER = 1


def load_dataset(augment: bool = False):
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=90,
            validation_split=0.2
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2
        )

    # Train data
    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='binary',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    # Validation data
    valid_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='binary',
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    # Test data
    if os.path.exists(TEST_DIR):
        test_generator = ImageDataGenerator(
            rescale=1. / 255
        ).flow_from_directory(
            directory=TEST_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            class_mode='binary',
            batch_size=BATCH_SIZE
        )
    else:
        test_generator = None

    print("Labels:", train_generator.class_indices)
    return train_generator, valid_generator, test_generator


def compose_model(activation: str = 'relu', padding: str = 'valid'):
    # Compose model structure
    _model = Sequential()

    _model.add(Conv2D(4, (3, 3), input_shape=INPUT_SHAPE, activation=activation, padding=padding))
    _model.add(MaxPooling2D(pool_size=(2, 2), padding=padding))

    _model.add(Conv2D(8, (3, 3), activation=activation, padding=padding))
    _model.add(MaxPooling2D(pool_size=(2, 2), padding=padding))

    _model.add(Conv2D(16, (3, 3), activation=activation, padding=padding))
    _model.add(MaxPooling2D(pool_size=(2, 2), padding=padding))

    _model.add(Flatten())

    _model.add(Dense(units=64, activation=activation))
    _model.add(Dropout(0.75))

    _model.add(Dense(units=1, activation='sigmoid'))

    _model.summary()
    return _model


def lr_scheduler(epoch, lr):
    if epoch < 10 or lr < 0.00005:
        return lr
    else:
        return lr * math.exp(-0.1)


def plot_graphs(_history):
    # Plot training & validation accuracy values
    plt.plot(_history.history['acc'])
    plt.plot(_history.history['val_acc'])
    plt.title(f'Model accuracy\nBATCH_SIZE = {BATCH_SIZE}, MULTIPLIER = {STEPS_MULTIPLIER}, LR = {LEARNING_RATE},\n'
              f'TRAIN_SAMPLES = {train_data.samples}, VALIDATION_SAMPLES = {validation_data.samples}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(_history.history['loss'])
    plt.plot(_history.history['val_loss'])
    plt.title(f'Model loss\nBATCH_SIZE = {BATCH_SIZE}, MULTIPLIER = {STEPS_MULTIPLIER}, LR = {LEARNING_RATE},\n'
              f'TRAIN_SAMPLES = {train_data.samples}, VALIDATION_SAMPLES = {validation_data.samples}')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# START ################################################################################################################

# Load dataset from disk
train_data, validation_data, test_data = load_dataset()

# Create and compile the model
model = compose_model(padding='same')
optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Save png representation of the model
plot_model(model, to_file='cnn_model.png', show_shapes=True)

# Setup callbacks to call
callbacks = [
    # EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    LearningRateScheduler(schedule=lr_scheduler, verbose=1)
]

# Train the model
start = datetime.datetime.now()
history = model.fit_generator(
    generator=train_data,
    epochs=EPOCHS,
    steps_per_epoch=(train_data.samples / BATCH_SIZE) * STEPS_MULTIPLIER,
    validation_data=validation_data,
    validation_steps=(validation_data.samples / BATCH_SIZE) * STEPS_MULTIPLIER,
    verbose=2,
    callbacks=callbacks
)
end = datetime.datetime.now()
print('\nTime elapsed:', end - start)

# Evaluate the model
if test_data is not None:
    test_loss = model.evaluate_generator(generator=test_data)
    print(f"{model.metrics_names[0]}: {test_loss[0]}")
    print(f"{model.metrics_names[1]}: {test_loss[1]}")

# Plot accuracy and loss graphs
plot_graphs(history)
