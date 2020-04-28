import os
import datetime
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

# Parameters
IMG_SIZE = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
DIR_TRAIN = '../dataset/train'
DIR_VALID = '../dataset/validation'
DIR_TEST = '../dataset/test'
EPOCHS = 10
BATCH_SIZE = 16

# Data pre-processing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    directory=DIR_TRAIN,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE
)

test_generator = test_datagen.flow_from_directory(
    directory=DIR_TEST,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE
)

valid_generator = valid_datagen.flow_from_directory(
    directory=DIR_VALID,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE
)

print(train_generator.class_indices)

# Model structure
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# needs pydot, graphviz
# plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)

start = datetime.datetime.now()
model.fit_generator(
    generator=train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator
)

end = datetime.datetime.now()
print('\nTime elapsed:', end-start)

test_loss = model.evaluate_generator(
    generator=test_generator
)

print(f"{model.metrics_names[0]}: {test_loss[0]}")
print(f"{model.metrics_names[1]}: {test_loss[1]}")
