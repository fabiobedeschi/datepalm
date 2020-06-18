import datetime
import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.utils.vis_utils import plot_model

from src.config import TRAIN_DIR, TEST_DIR, LEARNING_RATE, EPOCHS, BATCH_SIZE, STEPS_MULTIPLIER

from src.classifier.dataset import load_dataset
from src.classifier.model import compose_model, lr_scheduler
from src.classifier.plotter import plot_graphs

# START ################################################################################################################

# Load dataset from disk
train_data, validation_data, test_data = load_dataset(train_dir=TRAIN_DIR, test_dir=TEST_DIR)

# Create the model
model = compose_model(padding='valid')

# Compile the model
optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Save png representation
plot_model(model, to_file='cnn_model.png', show_shapes=True)

# Setup callbacks to call
callbacks = [
    # EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    LearningRateScheduler(schedule=lr_scheduler, verbose=1)
]

# Train the model
start = datetime.datetime.now()
history = model.fit_generator(
    generator=train_data,
    epochs=EPOCHS,
    steps_per_epoch=(train_data.samples / BATCH_SIZE) * STEPS_MULTIPLIER,
    validation_data=validation_data,
    validation_steps=(validation_data.samples / BATCH_SIZE) * STEPS_MULTIPLIER,
    verbose=2,
    callbacks=callbacks
)
end = datetime.datetime.now()
print('\nTime elapsed:', end - start)

# Evaluate the model
test_loss = model.evaluate_generator(generator=test_data)
print(f"{model.metrics_names[0]}: {test_loss[0]}")
print(f"{model.metrics_names[1]}: {test_loss[1]}")

# Plot accuracy and loss graphs
plot_graphs(history=history,
            train_samples=train_data.samples,
            validation_samples=validation_data.samples,
            test_acc=test_loss[1],
            test_loss=test_loss[0])
