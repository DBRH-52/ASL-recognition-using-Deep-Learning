# app_laptop.py
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize the webcam and hand detection module
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Initialize the hand classification module
classifier = Classifier("Model/asl_model.h5", "Model/labels.txt")

# Set parameters for image processing
offset = 20
image_size = 300
labels = ["A", "K"]

while True:
    # Read a frame from the webcam
    success, image = cap.read()
    output_image = image.copy()

    # Detect hands in the frame
    hands, image = detector.findHands(image)

    if hands:
        # Get the first detected hand
        hand = hands[0]
        x, y, width, height = hand['bbox']

        # Create a white background image
        background_image = np.ones((image_size, image_size, 3), np.uint8) * 255 #255, 225, 225 - RGB - white

        # Crop the region around the hand with an offset
        crop_image = image[y - offset:y + height + offset, x - offset:x + width + offset]
        ratio = height / width

        # Resize the cropped image to fit the white background
        if ratio > 1:
            k = image_size / height
            calculated_width = math.ceil(k * width)
            resize_image = cv2.resize(crop_image, (calculated_width, image_size))
            background_image[:, :calculated_width] = resize_image
        else:
            k = image_size / width
            calculated_height = math.ceil(k * height)
            resize_image = cv2.resize(crop_image, (image_size, calculated_height))
            background_image[:calculated_height, :] = resize_image

        # Get the hand gesture prediction using the trained classifier
        prediction, index = classifier.getPrediction(background_image, draw=False)

        # Draw rectangles and text on the output image
        cv2.rectangle(output_image, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(output_image, labels[index], (x, y-26), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(output_image, (x-offset, y-offset),
                      (x + width+offset, y + height+offset), (255, 0, 255), 4)

        # Display the cropped hand image and the white background image
        cv2.imshow("ImageCrop", crop_image)
        cv2.imshow("ImageWhite", background_image)

    # Display the output image with annotations
    cv2.imshow("Image", output_image)

    # Wait for a key press to exit the loop
    cv2.waitKey(1)
