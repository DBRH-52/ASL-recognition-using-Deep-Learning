# model_training.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load prepared data
train_data = np.load('train_data.npy') / 255.0
train_labels = np.load('train_labels.npy')

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')  # 26 classes (A-Z)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Save the trained model
model.save('asl_model.h5')


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
