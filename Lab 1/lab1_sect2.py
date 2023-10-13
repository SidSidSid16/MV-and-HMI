#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:30:11 2023

@author: sid
"""

import cv2
import numpy as np

img = cv2.imread('/Users/sid/Documents/Uni/Course/Year 4/Machine Vision and Human Machine Interaction/Lab 1 Part 1/gradient.png')

cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyWindow("Image")

