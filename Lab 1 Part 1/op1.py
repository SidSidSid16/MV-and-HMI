import os
import cv2

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Video Stream")

def blueCircleDetect():
    success, img = cap.read()
    params = cv2.SimpleBlobDetector_Params()
    params.thresholdStep = 255
    params.minRepeatability = 1
    params.blobColor = 255
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByArea = False
    params.filterByInertia = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)

    while success and cv2.waitKey(1) == -1:
        success, img = cap.read()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blueMin = (100, 100, 0)
        blueMax = (140, 255, 255)
        mask = cv2.inRange(hsv, blueMin, blueMax)
        keypoints = detector.detect(mask)
        imagekp = cv2.drawKeypoints(mask, keypoints, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Video Stream", imagekp)
    
    cv2.destroyWindow("Video Stream")
    cap.release()


def blueRectDetect():
    success, img = cap.read()
    params = cv2.SimpleBlobDetector_Params()
    params.thresholdStep = 255
    params.minRepeatability = 1
    params.blobColor = 255
    params.filterByColor = False
    params.filterByArea = False
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.maxCircularity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
    params.maxInertiaRatio = 0.15
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)

    while success and cv2.waitKey(1) == -1:
        success, img = cap.read()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blueMin = (100, 100, 0)
        blueMax = (140, 255, 255)
        mask = cv2.inRange(hsv, blueMin, blueMax)
        keypoints = detector.detect(mask)
        imagekp = cv2.drawKeypoints(mask, keypoints, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Video Stream", imagekp)
    
    cv2.destroyWindow("Video Stream")
    cap.release()


def redStarDetect():
    success, img = cap.read()
    params = cv2.SimpleBlobDetector_Params()
    params.thresholdStep = 255
    params.minRepeatability = 1
    params.blobColor = 255
    params.filterByArea = False
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.maxCircularity = 0.4
    params.filterByInertia = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)

    while success and cv2.waitKey(1) == -1:
        success, img = cap.read()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(hsv, (175,50,20), (180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)
        keypoints = detector.detect(mask)
        imagekp = cv2.drawKeypoints(mask, keypoints, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Video Stream", imagekp)
    
    cv2.destroyWindow("Video Stream")
    cap.release()

# blueCircleDetect()
# blueRectDetect()
redStarDetect()