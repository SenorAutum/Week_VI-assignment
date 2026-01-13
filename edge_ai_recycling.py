import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Data Preparation (Simulation) ---
# NOTE: In a real scenario, you would upload a folder of images.
# For this assignment to be runnable immediately, we will use the CIFAR-10 dataset 
# but we will 'pretend' specific classes are recyclable items for the prototype.
# Class mapping simulation: 
# 0: Airplane -> "Metal Can"
# 1: Automobile -> "Plastic Bottle"
# 8: Ship -> "Glass Jar"

print("Loading dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Filter dataset to just 3 classes to simulate our specific "Recyclables" problem
relevant_classes = [0, 1, 8] 
mask_train = np.isin(train_labels, relevant_classes).flatten()
mask_test = np.isin(test_labels, relevant_classes).flatten()

x_train = train_images[mask_train] / 255.0 # Normalize pixel values
y_train = train_labels[mask_train]
x_test = test_images[mask_test] / 255.0
y_test = test_labels[mask_test]

# Remap labels to 0, 1, 2 for our new specific task
def remap_labels(labels):
    new_labels = labels.copy()
    new_labels[labels == 0] = 0 # Metal
    new_labels[labels == 1] = 1 # Plastic
    new_labels[labels == 8] = 2 # Glass
    return new_labels

y_train = remap_labels(y_train)
y_test = remap_labels(y_test)

print(f"Training data shape: {x_train.shape}")
print("Classes: Metal (0), Plastic (1), Glass (2)")

# --- 2. Model Training (Lightweight CNN) ---
# We use a simple CNN suitable for Edge devices (low parameter count)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 output classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nStarting training...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# --- 3. Convert to TensorFlow Lite (The "Edge" Step) ---
print("\nConverting model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# OPTIONAL: Apply optimization (Quantization) to reduce model size further
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the model
tflite_model_path = 'recycling_model_quantized.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Success! Model saved to {tflite_model_path}")
print(f"Model Size: {os.path.getsize(tflite_model_path) / 1024:.2f} KB")

# --- 4. Deployment Check (Inference Test) ---
# Test the TFLite model on a single image to ensure it works
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test on the first image from test set
test_image = x_test[0:1].astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print(f"\nTest Inference Result: Class {np.argmax(output_data)}")
