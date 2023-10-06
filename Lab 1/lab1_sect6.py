# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:40:13 2023

@author: ss2985
"""

import cv2

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Video Stream")
success, img = cap.read()

blueMin = (100, 100, 0)
blueMax = (140, 255, 255)

params = cv2.SimpleBlobDetector_Params()
params.thresholdStep = 255
params.minRepeatability = 1
params.blobColor = 255
params.filterByColor = False
params.filterByArea = False
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)

while success and cv2.waitKey(1) == -1:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blueMin, blueMax)
    keypoints = detector.detect(mask)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    imagekp = cv2.drawKeypoints(masked_img, keypoints, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # masked_img = cv2.bitwise_and(img, img, mask=mask)
    # imagekp = cv2.drawKeypoints(mask, keypoints, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Video Stream", imagekp)
    success, img = cap.read()

cv2.destroyWindow("Video Stream")
cap.release()