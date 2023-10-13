import cv2
import cv2.aruco as aruco


ARUDO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMETERS = aruco.DetectorParameters()

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Video Stream")

success, img = cap.read()

while success and cv2.waitKey(1) == -1:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUDO_DICT, parameters=ARUCO_PARAMETERS)
    img = aruco.drawDetectedMarkers(img, corners, borderColor=(0,0,255))
    cv2.imshow("Video Stream", img)
    success, img = cap.read()

cv2.destroyWindow("Video Stream")
cap.release()