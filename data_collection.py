# data_collection.py
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Function to process and save the image
def process_and_save_image(image, counter, letter_folder, offset, image_size):
    # Find hands in the current frame
    hands, image = hand_detector.findHands(image)

    if hands:
        # Select the first detected hand (left or right)
        hand = hands[0]
        x, y, width, height = hand['bbox']
        background_image = np.ones((image_size, image_size, 3), np.uint8) * 255
        crop_image = image[y - offset:y + height + offset, x - offset:x + width + offset]

        # Image resizing logic
        ratio = height / width
        if ratio > 1:
            k = image_size / height
            calculated_width = math.ceil(k * width)
            resize_image = cv2.resize(crop_image, (calculated_width, image_size))
            width_gap = math.ceil((image_size - calculated_width) / 2)
            background_image[:, width_gap:calculated_width + width_gap] = resize_image
        else:
            k = image_size / width
            calculated_height = math.ceil(k * height)
            resize_image = cv2.resize(crop_image, (image_size, calculated_height))
            height_gap = math.ceil((image_size - calculated_height) / 2)
            background_image[height_gap:calculated_height + height_gap, :] = resize_image

        # Display the cropped and resized hand image
        cv2.imshow("CropImage", crop_image)
        cv2.imshow("BackgroundImage", background_image)

        # Save the processed image with a unique filename
        cv2.imwrite(f'{letter_folder}/Image_{time.time()}.jpg', background_image)
        print(counter)

offset = 20
image_size = 64
letter_folder = "Letters/A"
counter = 0

capture_video = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)

while True:
    # Capture video frames
    success, image = capture_video.read()

    # Process and save image on 'c' key press
    key = cv2.waitKey(1)
    if key == ord("c"):
        counter += 1
        process_and_save_image(image, counter, letter_folder, offset, image_size)

    # Display the original frame
    cv2.imshow("Image", image)

    # Exit on 'q' key press
    if key == ord("q"):
        break

# Releasing the video capture object and close windows
capture_video.release()
cv2.destroyAllWindows()
