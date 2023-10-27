import cv2

import os
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW, (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))

# mtx = np.genfromtxt('intrinsic_matrix.csv', delimiter=',')
# dist = np.genfromtxt('distortion_coeff.csv', delimiter=',')
# rvecs = np.genfromtxt('rotation_vect.csv', delimiter=',')
# tvecs = np.genfromtxt('translation_vect.csv', delimiter=',')

# Create a VideoCapture instance
cap = cv2.VideoCapture(1)
# Check if the deviceo could be opened and exit if not
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set the resolution for the image
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set up windows for the video stream and captured images
cv2.namedWindow("Video Stream")
cv2.namedWindow("Captured Image")

# Initialise an empty list of images and the number to be captured
number_of_images = 2
imglist = []

success = True

objpoints = []
imgpoints = []
imgdescs = []
# Loop through the indices of images to be captured
for imgnum in range(number_of_images):
    # Capture images continuously and wait for a key press
    while success and cv2.waitKey(1) == -1:
        # Read an image from the VideoCapture instance
        success, img = cap.read()
        cv2.imshow("Video Stream", img)
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        img_orb = cv2.drawKeypoints(img, kp, None, flags=0)
    
    
    imglist.append(img)
    imgpoints.append(kp)
    imgdescs.append(des)
    cv2.imshow("Captured Image", img_orb)
    print("Image Captured")
        
    if (len(imglist) > 1):
        gray1 = cv2.cvtColor(imglist[-1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(imglist[-2], cv2.COLOR_BGR2GRAY)
        img1 = imglist[-1]
        img2 = imglist[-2]
        kp1 = imgpoints[-1]
        kp2 = imgpoints[-2]
        des1 = imgdescs[-1]
        des2 = imgdescs[-2]
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Captured Image", img_matches)
        cv2.waitKey(0)
        
        
        

# Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")
cap.release()
