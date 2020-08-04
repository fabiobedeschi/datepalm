import os

import numpy as np

from src.autoencoder.params import BATCH_SIZE

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import matplotlib.pyplot as plt

from src.config import AED_TEST_DIR, AED_TRAIN_DIR
from src.autoencoder.dataset import load_test_dataset

from keras.models import load_model

train_generator = load_test_dataset(test_dir=AED_TRAIN_DIR)
test_generator = load_test_dataset(test_dir=AED_TEST_DIR)

model_code = '2020-07-31T20:44'
model = load_model(f'./models/{model_code}.h5')

test = model.evaluate_generator(generator=train_generator, steps=len(train_generator))

print(f"{model.metrics_names[0]}: {test[0]}")
# print(f"{model.metrics_names[1]}: {test[1]}")

# test_pred_data = model.predict_generator(generator=test_data, steps=BATCH_SIZE)
#
# print(test_pred_data)

# Test the model by viewing a sample of original and reconstructed images
data_list = []
batch_index = 0
while batch_index <= train_generator.batch_index:
    data = next(train_generator)
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
plt.show()

train_x = np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
train_y = np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])
train_p = model.predict_generator(generator=train_generator, steps=len(train_generator))

test_x = np.concatenate([test_generator.next()[0] for i in range(test_generator.__len__())])
test_y = np.concatenate([test_generator.next()[1] for i in range(test_generator.__len__())])
test_p = model.predict_generator(generator=test_generator, steps=len(test_generator))

train_mse = np.mean(np.power(train_x - train_p, 2), axis=(1, 2, 3))
print(np.mean(np.power(train_x - train_p, 2)))
print((train_mse < 0.017).sum())

test_mse = np.mean(np.power(test_x - test_p, 2), axis=(1, 2, 3))
print(np.mean(np.power(test_x - test_p, 2)))
print((test_mse < 0.017).sum())

colors = ['#FF0000' if c == 1 else '#00FF00' for c in test_generator.classes]
fig, axs = plt.subplots(2)
fig.suptitle('Mean Squared Errors')
axs[0].scatter(range(len(train_mse)), train_mse, c='#00FF00')
axs[0].set_title('Train')
axs[0].set_ylabel('MSE')
axs[1].scatter(range(len(test_mse)), test_mse, c=colors)
axs[1].set_title('Test')
axs[1].set_ylabel('MSE')
plt.show()
