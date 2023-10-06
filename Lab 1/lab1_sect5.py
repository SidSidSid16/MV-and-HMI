# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:40:13 2023

@author: ss2985
"""

import cv2
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Video Stream")
success, img = cap.read()

blueMin = (0, 0, 200)
blueMax = (100, 90, 200)

while success and cv2.waitKey(1) == -1:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blueMin, blueMax)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Video Stream", masked_img)
    success, img = cap.read()

cv2.destroyWindow("Video Stream")
cap.release()