import os

from src.autoencoder.params import BATCH_SIZE

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import matplotlib.pyplot as plt

from src.config import AED_TEST_DIR, AED_TRAIN_DIR
from src.autoencoder.dataset import load_test_dataset

from keras.models import load_model

data_generator = load_test_dataset(test_dir=AED_TEST_DIR)

model_code = '2020-07-30T11:38'
model = load_model(f'./models/{model_code}.h5')

test = model.evaluate_generator(generator=data_generator, steps=len(data_generator))

print(f"{model.metrics_names[0]}: {test[0]}")
print(f"{model.metrics_names[1]}: {test[1]}")

# test_pred_data = model.predict_generator(generator=test_data, steps=BATCH_SIZE)
#
# print(test_pred_data)

# Test the model by viewing a sample of original and reconstructed images
data_list = []
batch_index = 0
while batch_index <= data_generator.batch_index:
    data = next(data_generator)
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
