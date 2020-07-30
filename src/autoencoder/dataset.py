from keras_preprocessing.image import ImageDataGenerator

from src.config import IMG_SIZE
from src.autoencoder.params import BATCH_SIZE


def load_train_dataset(train_dir: str, batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=False, val_split=0.2):
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            height_shift_range=0.2,
            width_shift_range=0.2,
            fill_mode='constant',
            validation_split=val_split
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=val_split
        )

    # Train data
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(img_size, img_size),
        class_mode='input',
        batch_size=batch_size,
        subset='training'
    )

    # Validation data
    valid_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(img_size, img_size),
        class_mode='input',
        batch_size=batch_size,
        subset='validation'
    )

    return train_generator, valid_generator


def load_test_dataset(test_dir: str, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    test_generator = ImageDataGenerator(
        rescale=1. / 255
    ).flow_from_directory(
        directory=test_dir,
        target_size=(img_size, img_size),
        class_mode='input',
        batch_size=batch_size
    )

    return test_generator
