import numpy as np
import cv2

import os
os.environ['OPENCV_VIDEOIO_MSMF_EMABLE_HW_TRANSFORMS'] = '0'

mtx = np.genfromtxt('Lab 2/intrinsic_matrix.csv', delimiter=',')
dist = np.genfromtxt('Lab 2/distortion_coeff.csv', delimiter=',')

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW, (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Video Stream")
cv2.namedWindow("Captured Image")

number_of_images = 2
imglist = []

success = True
for imgnum in range(number_of_images):
    while success and cv2.waitKey(1) == -1:
        success, img = cap.read()
        cv2.imshow("Video Stream", img)
    imglist.append(img)
    cv2.imshow("Captured Image", img)
    print("Image Captured")

cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undist_imglist = []
x, y, w, h = roi
for img in imglist:
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    dst = dst[y:y+h, x:x+w]
    undist_imglist.append(dst)

objpoints = []
imgpoints = []
imgdescs = []
orb = cv2.ORB_create()
for img in undist_imglist:
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    imglist.append(img)
    imgpoints.append(kp)
    imgdescs.append(des)

img_matches_list = []
for img in undist_imglist:
    if (len(undist_imglist) > 1):
        gray1 = cv2.cvtColor(undist_imglist[-1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(undist_imglist[-2], cv2.COLOR_BGR2GRAY)
        img1 = undist_imglist[-1]
        img2 = undist_imglist[-2]
        kp1 = imgpoints[-1]
        kp2 = imgpoints[-2]
        des1 = imgdescs[-1]
        des2 = imgdescs[-2]
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        img_matches_list.append(matches)

        pts1 = []
        pts2 = []
        for m in matches:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        pts2 = np.int32(pts2)
        pts1 = np.int32(pts1)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.LMEDS, prob=0.999)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

print("Fundamental mtx", F)
print("Essential mtx", E)
print("Homography mtx", H)

cv2.namedWindow("Stitched Image")
warped = cv2.warpPerspective(undist_imglist[0], H, (undist_imglist[1].shape[1], undist_imglist[1].shape[0]))
cv2.imshow("Stitched Image", undist_imglist[1]+warped)
cv2.waitKey(0)
cv2.destroyWindow("Stitched Stream")
