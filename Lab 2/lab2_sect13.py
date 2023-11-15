import numpy as np
import cv2

import os
os.environ['OPENCV_VIDEOIO_MSMF_EMABLE_HW_TRANSFORMS'] = '0'

mtx = np.genfromtxt('Lab 2/intrinsic_matrix.csv', delimiter=',')
dist = np.genfromtxt('Lab 2/distortion_coeff.csv', delimiter=',')

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Video Stream")
cv2.namedWindow("Captured Image")

imglist = []

success = True


while True:
    success, img = cap.read()
    cv2.imshow("Video Stream", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        captured_img = img
        imglist.append(captured_img)
        cv2.imshow("Captured Image", img)
        print("Image Captured")
    elif key == ord('q'):
        break

cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")
cap.release()

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

if len(imglist) >= 2:
    # Use the first image as the base
    panorama = imglist[0]
    for i in range(1, len(imglist)):
        img1_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(imglist[i], cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.15)]

        if len(good_matches) >= 10:
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # dimensions of panorama
            h1, w1 = panorama.shape[:2]
            # dimensions of image to stitch
            h2, w2 = imglist[i].shape[:2]
            # corners of panorama
            pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            # corners of image to stitch
            pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            # transform pts2 using homography to give new corners
            pts2_ = cv2.perspectiveTransform(pts2, H)
            # combine the corners 
            pts = np.concatenate((pts1, pts2_), axis=0)
            # min and max size of the canvas
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
            # translation vector to move stitched image so there are no negative coords.
            t = [-xmin, -ymin]
            # homography matrix applying translation
            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # Translation matrix

            # image is warped with the combined homography matrix.
            # warps the image as per H and translates as per Ht
            result = cv2.warpPerspective(imglist[i], Ht.dot(H), (xmax-xmin, ymax-ymin))
            # current panorama is copied onto the new canvas
            result[t[1]:h1+t[1], t[0]:w1+t[0]] = panorama
            panorama = result  # Update panorama

cv2.namedWindow("Panorama")
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undist_imglist = []
# x, y, w, h = roi
# for img in imglist:
#     dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#     dst = dst[y:y+h, x:x+w]
#     undist_imglist.append(dst)

# objpoints = []
# imgpoints = []
# imgdescs = []
# orb = cv2.ORB_create()
# for img in undist_imglist:
#     kp = orb.detect(img, None)
#     kp, des = orb.compute(img, kp)
#     imglist.append(img)
#     imgpoints.append(kp)
#     imgdescs.append(des)

# height, width = undist_imglist[0].shape[:2]
# blank_image = np.zeros((3840,2160,3), np.uint8)
# blank_image[:,:] = (255,255,255)
# l_img = blank_image.copy()
# x_offset = (int) (3840/2-1280/2)
# y_offset = (int) (2160/2-720/2)
# l_img[y_offset:y_offset+height, x_offset:x_offset+width] = undist_imglist[0].copy()

# img_matches_list = []
# for idx,img in enumerate(undist_imglist):
#     if (len(undist_imglist) > 1):

#         kp = orb.detect(img, None)
#         kp2, des2 = orb.compute(img, kp)

#         img1 = undist_imglist[idx]
#         img2 = l_img
#         kp1 = imgpoints[idx]
#         des1 = imgdescs[idx]
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         matches = bf.match(des1, des2)
#         matches = sorted(matches, key = lambda x:x.distance)
#         img_matches_list.append(matches)

#         pts1 = []
#         pts2 = []
#         for m in matches:
#             pts2.append(kp2[m.trainIdx].pt)
#             pts1.append(kp1[m.queryIdx].pt)
#         pts2 = np.int32(pts2)
#         pts1 = np.int32(pts1)
#         F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
#         E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.LMEDS, prob=0.999)
#         _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)
#         H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
#         warped = cv2.warpPerspective(img1, H, (l_img.shape[1], l_img.shape[0]))
#         alpha = 0.5
#         l_img = cv2.addWeighted(warped, alpha, l_img, 1 - alpha, 0)

# print("Fundamental mtx", F)
# print("Essential mtx", E)
# print("Homography mtx", H)

# cv2.namedWindow("Stitched Image")
# # warped = cv2.warpPerspective(undist_imglist[0], H, (l_img.shape[1], l_img.shape[0]))
# # alpha = 0.5
# # blended = cv2.addWeighted(warped, alpha, undist_imglist[1], 1 - alpha, 0)
# cv2.imshow("Stitched Image", l_img)
# cv2.waitKey(0)
# cv2.destroyWindow("Stitched Image")
