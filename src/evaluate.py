import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = tf.keras.models.load_model("models/fashion_mnist_ann.h5")

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

np.save("models/confusion_matrix.npy", cm)

print("Evaluation completed successfully.")
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Confusion matrix saved.")
