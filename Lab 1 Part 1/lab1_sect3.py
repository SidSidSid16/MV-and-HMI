#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:36:38 2023

@author: sid
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Video Stream")
success, img = cap.read()

while success and cv2.waitKey(1) == -1:
	cv2.imshow("Video Stream", img)
	success, img = cap.read()

cv2.destroyWindow("Video Stream")
cap.release()