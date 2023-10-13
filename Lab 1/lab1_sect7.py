import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Video Stream")
success, img = cap.read()

while success and cv2.waitKey(1) == -1:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray, (9,9), 0)
    cv2.imshow("Video Stream", blur_img)
    success, img = cap.read()

cv2.destroyWindow("Video Stream")
cap.release()