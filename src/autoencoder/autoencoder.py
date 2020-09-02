import os
from datetime import datetime

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam

from src.config import AED_TRAIN_DIR
from src.plotter import plot_graphs

from src.autoencoder.params import *
from src.autoencoder.dataset import load_train_dataset
from src.autoencoder.model import compose_model, lr_scheduler
from src.autoencoder.prediction import prediction

# START ################################################################################################################
session_id = datetime.now().isoformat()[:16]

# Load dataset from disk
train_data, validation_data = load_train_dataset(train_dir=AED_TRAIN_DIR, val_split=0.25)

# Create the model
model = compose_model(filters=[16, 20, 24, 28, 32])

# Compile the model
loss_func = 'mse'
optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss=loss_func, optimizer=optimizer)

# Save png representation
plot_model(model, to_file=f'./models/{session_id}.png', show_shapes=True)

# Setup callbacks to call
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    TensorBoard(log_dir=f'./logs/{session_id}', batch_size=BATCH_SIZE),
    LearningRateScheduler(schedule=lr_scheduler, verbose=1)
]

# Train the model
start = datetime.now()
history = model.fit_generator(
    generator=train_data,
    epochs=EPOCHS,
    steps_per_epoch=len(train_data) * STEPS_MULTIPLIER,
    validation_data=validation_data,
    validation_steps=len(validation_data) * STEPS_MULTIPLIER,
    verbose=2,
    callbacks=callbacks
)
end = datetime.now()
print('\nTime elapsed:', end - start)

# Save the model
model.save(f'./models/{session_id}.h5')

# Plot accuracy and loss graphs
plot_graphs(
    history=history,
    batch_size=BATCH_SIZE,
    train_samples=train_data.samples,
    validation_samples=validation_data.samples,
    session_id=session_id
)

