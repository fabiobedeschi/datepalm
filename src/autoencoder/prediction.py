import os

import numpy as np

from src.autoencoder.params import BATCH_SIZE

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import matplotlib.pyplot as plt

from src.config import AED_TEST_DIR, AED_TRAIN_DIR
from src.autoencoder.dataset import load_test_dataset

from keras.models import load_model

generators = {'train': load_test_dataset(test_dir=AED_TRAIN_DIR), 'test': load_test_dataset(test_dir=AED_TEST_DIR)}
threshold = 0.017

model_code = '2020-07-31T20:44'
model = load_model(f'./models/{model_code}.h5')

for gen_name, generator in generators.items():
    print(gen_name)

    evaluation = model.evaluate_generator(generator=generator, steps=len(generator))

    print(f"{model.metrics_names[0]}: {evaluation[0]}")
    print(f"{model.metrics_names[1]}: {evaluation[1]}")

    # Test the model by viewing a sample of original and reconstructed images
    data_list = []
    batch_index = 0
    while batch_index <= generator.batch_index:
        data = next(generator)
        data_list.append(data[0])
        batch_index = batch_index + 1

    predicted = model.predict(data_list[0])
    no_of_samples = 4
    _, axs = plt.subplots(no_of_samples, 2, figsize=(5, 8))
    axs = axs.flatten()
    imgs = []
    for i in range(no_of_samples):
        imgs.append(data_list[i][i])
        imgs.append(predicted[i])
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.suptitle(f'{model_code} on {gen_name}')
    plt.show()

    samples = np.concatenate([next(generator)[0] for _ in range(len(generator))])
    predictions = model.predict_generator(generator=generator, steps=len(generator))

    se = np.power(np.subtract(predictions, samples), 2)
    mse = np.mean(se, axis=(1, 2, 3))
    # print(mse)
    print('mse:', np.mean(se))
    print('under threshold:', (mse < threshold).sum())

    colors = ['#FF0000' if c == 1 else '#00FF00' for c in generator.classes]
    plt.scatter(range(len(mse)), mse, c=colors)
    plt.axhline(y=threshold, color='#E0E000', linestyle='-')
    plt.title(f'{model_code} on {gen_name}')
    plt.ylabel('MSE')
    plt.show()
    print()
