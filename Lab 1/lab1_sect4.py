# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:36:09 2023

@author: ss2985
"""

import cv2

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"


img = cv2.imread('gradient.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh_value,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
thresh_value,thresh2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
thresh_value,thresh3 = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
thresh_value,thresh4 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
thresh_value,thresh5 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    cv2.namedWindow(titles[i])
    cv2.imshow(titles[i], images[i])
    cv2.waitKey(0)
    cv2.destroyWindow(titles[i])
