# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import mnist

# # Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize and reshape data
# x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)

# # Build CNN model
# model = Sequential([
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')  # Output layer
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# # Save the model
# model.save("cnn_digit_classifier.h5")


# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image, ImageOps

# # Load the trained CNN model
# model = load_model("cnn_digit_classifier.h5")

# # Streamlit app title and description
# st.title("Digit Classifier (CNN)")
# st.write("Upload a grayscale image of a handwritten digit (28x28 pixels).")

# # File uploader for the image
# uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Open and preprocess the uploaded image
#     image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     # Resize to 28x28 and normalize
#     image = ImageOps.fit(image, (28, 28), Image.LANCZOS)
#     image_array = np.array(image) / 255.0  # Normalize to [0, 1]
#     image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the CNN input

#     # Make predictions
#     prediction = model.predict(image_array)
#     predicted_digit = np.argmax(prediction)

#     # Display prediction and probabilities
#     st.write(f"Predicted Digit: **{predicted_digit}**")
#     st.bar_chart(prediction.flatten())  # Show probabilities as a bar chart
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build CNN model
model = Sequential([
    Input(shape=(28, 28, 1)),  # Explicitly defining input shape
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("cnn_digit_classifier.h5")

# -------------------- STREAMLIT APP --------------------
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained CNN model
model = load_model("cnn_digit_classifier.h5")

# Streamlit app title and description
st.title("Digit Classifier (CNN)")
st.write("Upload a grayscale image of a handwritten digit (28x28 pixels).")

# File uploader for the image
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and preprocess the uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to 28x28 and normalize
    image = ImageOps.fit(image, (28, 28), Image.LANCZOS)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the CNN input

    # Make predictions
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Display prediction and probabilities
    st.write(f"Predicted Digit: **{predicted_digit}**")
    st.bar_chart(prediction.flatten())  # Show probabilities as a bar chart
