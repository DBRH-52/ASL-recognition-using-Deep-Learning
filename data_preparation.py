# data_preparation.py
import os
import cv2
import numpy as np

def prepare_data(input_folder_path):
    # Lists to store image data and corresponding labels
    data = []
    labels = []

    # Iterate through each label (subdirectory) in the input folder
    for label in os.listdir(input_folder_path):
        label_path = os.path.join(input_folder_path, label)

        # Check if the current item is a directory
        if os.path.isdir(label_path):
            # Iterate through each image file in the current label's directory
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)

                # Read the image in grayscale and resize it to 64x64 pixels
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))  # Adjust size as needed

                # Append the image data and corresponding label to the lists
                data.append(img)
                
                # Convert label (alphabet letter) to a numeric value (0-25)
                labels.append(ord(label) - ord('A'))

    # Convert lists to NumPy arrays for better handling
    return np.array(data), np.array(labels)

# The path to the training data
folder_path = r'Data/Training/'
train_data, train_labels = prepare_data(folder_path)

# Save the NumPy arrays as binary files
np.save('train_data.npy', train_data)
np.save('train_labels.npy', train_labels)
