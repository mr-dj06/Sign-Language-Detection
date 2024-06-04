import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow 
import keras

offset = 20
imgSize = 300
counter = 0
labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

#folder = "Data\C"
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
           
        cv2.putText(imgOutput, labels[index], (x, y - 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255), 4)

        # cv2.imshow("Cropped",imgCrop)
        # cv2.imshow("White", imgWhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
    
