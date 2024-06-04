import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Create the main data folder if it does not exist
main_folder = "Data"
os.makedirs(main_folder, exist_ok=True)

# Set up video capture from webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0
num_images_per_label = 150

# Loop for each label
while True:
    # Prompt the user to enter a new label
    label = input(f"Enter label (A-Z, 0-9), '.' to quit: ").upper()
    if label == '.':
        break

    # Create a folder for the specified label within the main data folder
    folder = os.path.join(main_folder, label)
    os.makedirs(folder, exist_ok=True)

    # Loop for saving images for the current label
    for i in range(num_images_per_label):
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            break

        # Detect hands in the frame
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Create a white image of specified size
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Crop and resize the hand region
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display cropped and resized images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            # Save image
            counter += 1
            filename = f'{folder}/Image_{counter}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"Saved {filename}")

        # Display original frame
        cv2.imshow("Image", img)

        # Break the loop if '.' is pressed to change label
        if cv2.waitKey(1) == ord('.'):
            break

    # Break the outer loop if '.' is pressed to quit the program
    if label == '.':
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
