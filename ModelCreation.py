import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

import collections


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(12)

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'iloveyou', 'please', 'thanks', 'goodbye'])  # Add more if available
no_sequences = 100
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(1, no_sequences + 1):
        window = []
        for frame_num in range(sequence_length):
            path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            #print(path)
            window.append(np.load(path))
        sequences.append(window)
        labels.append(label_map[action])

print(collections.Counter(labels))

X = np.array(sequences)
y = to_categorical(labels).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12, stratify=y)

def add_noise(X, noise_factor=0.015):
    noise = noise_factor * np.random.randn(*X.shape)
    return X + noise

X_train = add_noise(X_train)


model = Sequential([
    LSTM(28, return_sequences=True, activation='relu', input_shape=(30, 1662), kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(28, return_sequences=False, activation='relu', input_shape=(30, 1662), kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.1),

    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dense(actions.shape[0], activation='softmax')
])


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


checkpoint = ModelCheckpoint(
    'best_model.h5',                  # file name
    monitor='val_categorical_accuracy',  # metric to monitor
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(monitor='val_categorical_accuracy',
                           patience=80,
                           mode='max',
                           restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy',
                              factor=0.5,
                              patience=18,
                              verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=80,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop, reduce_lr]
)

plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_accuracy_plot.png')
plt.show()

# Plot the loss graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot.png')
plt.show()
