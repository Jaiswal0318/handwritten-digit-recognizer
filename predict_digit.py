
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model(
    "models/digit_recognizer.keras"
)

# Load one image from the MNIST test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Prepare the image
image = x_test[0].astype("float32") / 255.0
image = image.reshape(1, 28, 28, 1)

# Predict the digit
predictions = model.predict(image, verbose=0)
predicted_digit = predictions.argmax()

print("Predicted digit:", predicted_digit)
print("Actual digit:", y_test[0])

# Display the image
plt.imshow(x_test[0], cmap="gray")
plt.title(f"Predicted: {predicted_digit} | Actual: {y_test[0]}")
plt.axis("off")
plt.show()
