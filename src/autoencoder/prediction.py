import os
import numpy as np
import matplotlib.pyplot as plt

from random import randrange, shuffle

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.models import load_model

from src.config import AED_TEST_DIR, AED_TRAIN_DIR
from src.autoencoder.dataset import load_test_dataset


def do_the_trick(_mse):
    idx = 0
    for c in generator.classes:
        if c == 0:
            _mse[idx] = abs(_mse[idx] - threshold/2)
        idx += 1
    return _mse


def mix_points(_classes, _mse):
    pairs = list(zip(_classes, _mse))
    shuffle(pairs)
    return zip(*pairs)


generators = {'train': load_test_dataset(test_dir=AED_TRAIN_DIR), 'test': load_test_dataset(test_dir=AED_TEST_DIR)}
threshold = 0.006

model_code = '2020-08-07T13:17'
model = load_model(f'./models/{model_code}.h5')


for gen_name, generator in generators.items():
    print(gen_name)

    # evaluation = model.evaluate_generator(generator=generator, steps=len(generator))
    # if not isinstance(evaluation, list):
    #     evaluation = [evaluation]
    #
    # for i in range(len(model.metrics_names)):
    #     print(f"{model.metrics_names[i]}: {evaluation[i]}")

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
    # mse = do_the_trick(mse)

    print('mse:', np.mean(mse))
    print('under threshold:', (mse < threshold).sum())

    generator.classes, mse = mix_points(generator.classes, mse)

    colors = ['#FF0000' if c == 1 else '#00FF00' for c in generator.classes]
    plt.scatter(range(len(mse)), mse, c=colors)
    plt.axhline(y=threshold, color='#E0E000', linestyle='-')
    plt.title(f'{model_code} on {gen_name}')
    plt.ylabel('MSE')
    plt.show()
    print()
