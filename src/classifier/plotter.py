import matplotlib.pyplot as plt

from src.config import BATCH_SIZE, STEPS_MULTIPLIER, LEARNING_RATE


def plot_graphs(history, train_samples, validation_samples, test_acc, test_loss):
    title_infos = f'BATCH_SIZE = {BATCH_SIZE}, MULTIPLIER = {STEPS_MULTIPLIER}, LR = {LEARNING_RATE},\n' \
                  f'TRAIN_SAMPLES = {train_samples}, VALIDATION_SAMPLES = {validation_samples},\n' \
                  f'test_acc = {round(test_acc, 2)}, test_loss = {round(test_loss, 2)}'

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(f'Model accuracy\n{title_infos}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss\n{title_infos}')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
