import os
import numpy as np
import matplotlib.pyplot as plt

from random import randrange, shuffle

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.models import load_model

from src.config import AED_TEST_DIR, AED_TRAIN_DIR
from src.autoencoder.dataset import load_test_dataset

_threshold = 0.00707
_model_code = '2020-08-12T14:28'
_model = load_model(f'./models/{_model_code}.h5')


def _mix_points(classes, mse):
    pairs = list(zip(classes, mse))
    shuffle(pairs)
    return zip(*pairs)


def prediction(model=_model, model_code=_model_code, threshold=_threshold):
    generators = {
        'train': load_test_dataset(test_dir=AED_TRAIN_DIR),
        'test': load_test_dataset(test_dir=AED_TEST_DIR)
    }

    for gen_name, generator in generators.items():
        print(gen_name)

        samples = np.concatenate([next(generator)[0] for _ in range(len(generator))])
        predictions = model.predict_generator(generator=generator, steps=len(generator))

        # Viewing a sample of original and reconstructed images
        no_of_samples = 4
        _, axs = plt.subplots(no_of_samples, 2, figsize=(5, 8))
        axs = axs.flatten()
        imgs = []
        for _ in range(no_of_samples):
            i = randrange(0, len(samples))
            imgs.append(samples[i])
            imgs.append(predictions[i])
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
        plt.suptitle(f'{model_code} on {gen_name}')
        plt.show()

        se = np.power(np.subtract(predictions, samples), 2)
        mse = np.mean(se, axis=(1, 2, 3))

        print('total mse:', np.mean(mse))
        print('total under threshold:', (mse < threshold).sum(), '/', generator.samples)
        if gen_name == 'test':
            print('good under threshold:',
                  sum(cls == 0 and val < threshold for cls, val in zip(generator.classes, mse)),
                  '/',
                  sum(cls == 0 for cls in generator.classes))

        generator.classes, mse = _mix_points(generator.classes, mse)

        colors = ['#FF0000' if c == 1 else '#00FF00' for c in generator.classes]
        plt.scatter(range(len(mse)), mse, c=colors, edgecolors='#40404040')
        plt.axhline(y=threshold, color='#E0E000C0', linestyle='-')
        plt.title(f'{model_code} on {gen_name}')
        plt.ylabel('MSE')
        plt.show()
        print()


prediction()
