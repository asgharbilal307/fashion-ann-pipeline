import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = tf.keras.models.load_model("models/model.h5")

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Fashion-MNIST Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

print("Evaluation completed successfully.")
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Confusion matrix image saved.")
