import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_params = params["train"]

hidden_units = train_params["hidden_units"]
dropout = train_params["dropout"]
learning_rate = train_params["learning_rate"]
epochs = train_params["epochs"]
batch_size = train_params["batch_size"]

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(hidden_units, activation="relu"),
    Dropout(dropout),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

model.save("models/model.h5")

history_df = pd.DataFrame(history.history)
history_df.to_csv("models/history.csv", index=False)

print("Training completed successfully.")
