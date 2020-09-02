import os
from datetime import datetime

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils.vis_utils import plot_model

from src.config import CNN_TRAIN_DIR, CNN_TEST_DIR
from src.plotter import plot_graphs

from src.classifier.params import *
from src.classifier.dataset import load_dataset
from src.classifier.model import compose_model, lr_scheduler

# START ################################################################################################################
session_id = datetime.now().isoformat()[:16]

# Load dataset from disk
train_data, validation_data, test_data = load_dataset(augment=True, train_dir=CNN_TRAIN_DIR, test_dir=CNN_TEST_DIR)

# Create the model
model = compose_model(filters=[20, 24, 28, 32, 36])

# Compile the model
loss_func = 'binary_crossentropy'
optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])

# Save png representation
plot_model(model, to_file=f'./models/{session_id}.png', show_shapes=True)

# Setup callbacks to call
callbacks = [
    # EarlyStopping(patience=15, verbose=1),
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

# Evaluate the model
test = model.evaluate_generator(generator=test_data, steps=len(test_data))
print(f"{model.metrics_names[0]}: {test[0]}")
print(f"{model.metrics_names[1]}: {test[1]}")

# Plot accuracy and loss graphs
plot_graphs(
    history=history,
    batch_size=BATCH_SIZE,
    train_samples=train_data.samples,
    validation_samples=validation_data.samples,
    session_id=session_id
)