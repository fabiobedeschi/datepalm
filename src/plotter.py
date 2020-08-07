import matplotlib.pyplot as plt


def plot_graphs(history, batch_size, train_samples, validation_samples, session_id, plot_dir='./plots'):
    title_infos = f'BATCH = {batch_size}, TRAIN = {train_samples}, VALIDATION = {validation_samples}'

    # Plot training & validation accuracy values
    if 'acc' in history.history:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(f'Model accuracy\n{title_infos}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        plt.savefig(f'{plot_dir}/{session_id}_accuracy.png')
        plt.show()

    # Plot training & validation loss values
    if 'loss' in history.history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model loss\n{title_infos}')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(f'{plot_dir}/{session_id}_loss.png')
        plt.show()
