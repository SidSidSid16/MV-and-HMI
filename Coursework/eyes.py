import cv2
import time
import math
import numpy as np
from consoledraw import Console
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

# Frame capture parameters
VIDEO_CAPTURE = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
# Path to shape predictor data
#   download the file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2,
#   extract the .dat file to a location, then paste in the full path of the .dat file in the
#   constant below
FACE_LANDMARK_DATA_PATH = "/full/path/to/shape_predictor_68_face_landmarks.dat"
# Blink detection parameters
BLINK_EAR_THRESHOLD = 0.4
BLINK_LOG_MAX_RECORDINGS = 600  # store a max of 10 mins worth of blink logs for 30 FPS footage
TIREDNESS_EAR_THRESHOLD = 0.35
# Gaze detection parameters
EYE_BININV_MAX_THRESHOLD = 255
EYE_BININV_MIN_THRESHOLD = 25

console = Console()
format = """
    Tot. frames: {}
    Avg. FPT: {} ms     (target: {} ms)
    Avg. FPS: {} fps    (target: {} fps)
    Eye tracking:
        Blink: {} (Avg. EAR: {})
        Blink score: {}

    {}
"""

# initialise video capture
capture = cv2.VideoCapture(VIDEO_CAPTURE)
# break program if capture cannot be opened
if not capture.isOpened():
    print("Cannot open video capture")
    exit()

# Initialise capture resolution
capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

# initialise windows for output
cv2.namedWindow("(a) Original")
cv2.namedWindow("(b) Grayscale")
cv2.namedWindow("(c) Histogram Equalisation")
cv2.namedWindow("(d) Gaussian Blur")
cv2.namedWindow("(e) Face Detection")
cv2.namedWindow("(f) Left Eye Detection")
cv2.namedWindow("(f) Right Eye Detection")

# initialise the array to store frame processing times
frame_processing_times = []

# get target FPS and FPT from input stream
capture_fps = round(capture.get(cv2.CAP_PROP_FPS), 0)
target_fpt = round((1/capture_fps) * 1000, 2)

# initialise models for landmark and face detection 
detector = dlib.get_frontal_face_detector() 
landmark_predict = dlib.shape_predictor(FACE_LANDMARK_DATA_PATH)
tiredness_score = 0

# initialise eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# initialise Simple Blob Detector parameters for iris\pupil tracking
sbd_params = cv2.SimpleBlobDetector_Params()
# disable thresholding process since input will already be grayscale
sbd_params.thresholdStep = 255
sbd_params.minRepeatability = 1
# detect blobs based on grayscale colour
sbd_params.filterByColor = True
sbd_params.blobColor = 255
# detect blobs by area
sbd_params.filterByArea = True
sbd_params.minArea = 100000
sbd_params.maxArea = 500000
# detect blobs based on circularity
sbd_params.filterByCircularity = False
sbd_params.minCircularity = 0.0001
sbd_params.maxCircularity = np.Inf
# detect blobs based on how diameter changes with angle
sbd_params.filterByInertia = True
sbd_params.minInertiaRatio = 0.0001
sbd_params.maxInertiaRatio = np.Inf
# detect blobs based on convexity
sbd_params.filterByConvexity = True
sbd_params.minConvexity = 0.0005
sbd_params.maxConvexity = np.Inf

# initialise Simple Blob Detector function
sbd_detector = cv2.SimpleBlobDetector_create(sbd_params)

# initialise eye monitoring data
blink_detected = False
EAR_alert = False
frame_EAR_log = []
frame_gaze_log = []
avg_EAR = 0
avg_gaze = 0

# capture first frame of stream
success, img = capture.read()

# track if feed is paused by user
paused = True

# continuously capture video feed
while success:
    # Get the dimensions of the frame
    frame_height, frame_width = img.shape[:2]

    # Calculate the center of the frame
    center_x, center_y = frame_width // 2, frame_height // 2

    # calculate the cropping boundaries based on the specified dimensions
    crop_start_x = max(center_x - (CAPTURE_WIDTH // 2), 0)
    crop_end_x = min(center_x + (CAPTURE_WIDTH // 2), frame_width)
    crop_start_y = max(center_y - (CAPTURE_HEIGHT // 2), 0)
    crop_end_y = min(center_y + (CAPTURE_HEIGHT // 2), frame_height)

    # adjust the cropping if it exceeds the frame boundaries
    if crop_end_x - crop_start_x < CAPTURE_WIDTH:
        if crop_end_x < frame_width:
            crop_end_x = min(crop_end_x + (CAPTURE_WIDTH - (crop_end_x - crop_start_x)), frame_width)
        else:
            crop_start_x = max(crop_start_x - (CAPTURE_WIDTH - (crop_end_x - crop_start_x)), 0)

    # perform the cropping
    img = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

    # Get key stroke
    key = cv2.waitKey(1)

    # exit if 'e' key pressed
    if key == ord('e'):
        break

    # enter paused mode with 'p' key
    if key == ord('p'):
        paused = True

    # record the start time of frame processing iteration
    frame_processing_start = time.time()

    # Display original video feed frame in window
    cv2.imshow("(a) Original", img)
    
    # Single-channel converstion using grayscale filter
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Display cropped video feed frame in window
    cv2.imshow("(b) Grayscale", grayscale)

    # Histogram equalisation to improve image contrast
    histequ = cv2.equalizeHist(grayscale)
    # Display histogram equalised frame in window
    cv2.imshow("(c) Histogram Equalisation", histequ)

    # Gaussian Blur to reduce noise and smoothening
    gaussian = cv2.GaussianBlur(histequ,(5,5),0)
    # Display blurred output frame in window
    cv2.imshow("(d) Gaussian Blur", gaussian)

    # detect faces using the dlib detector
    faces = detector(gaussian)

    # display the faces as boxes overlayed on the video feed
    for face in faces:
        # get face location coordinates
        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()
        
        # draw a box around detected face overlayed on video feed
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # detect facial landmarks
        face_landmarks = face_utils.shape_to_np(landmark_predict(gaussian, face))

        # parse landmark data to extract left and right eye landmarks
        left_eye = face_landmarks[L_start: L_end]
        right_eye = face_landmarks[R_start:R_end]

        # calculate height of each eye
        left_eye_y1 = dist.euclidean(left_eye[1], left_eye[5])
        left_eye_y2 = dist.euclidean(left_eye[2], left_eye[4])
        right_eye_y1 = dist.euclidean(right_eye[1], right_eye[5])
        right_eye_y2 = dist.euclidean(right_eye[2], right_eye[4])

        # calculate horizontal distance of each eye
        left_eye_x1 = dist.euclidean(left_eye[0], left_eye[3])
        right_eye_x1 = dist.euclidean(right_eye[0], right_eye[3])

        # calculate eye aspect ratio (EAR)
        left_EAR = (left_eye_y1 + left_eye_y2) / left_eye_x1
        right_EAR = (right_eye_y1 + right_eye_y2) / right_eye_x1
        avg_EAR = (left_EAR + right_EAR) / 2
        
        # log average EAR value
        frame_EAR_log.append(avg_EAR)

        # clear older logged EAR readings (garbage-collection of older recordings)
        if (len(frame_EAR_log) > BLINK_LOG_MAX_RECORDINGS):
            # get number of excess
            excess_vals = len(frame_EAR_log) - BLINK_LOG_MAX_RECORDINGS
            # clear the excess from the start of the list (clearing older excess values)
            del frame_EAR_log[:excess_vals]

        # blink detected if EAR falls below defined threshold
        if (avg_EAR < BLINK_EAR_THRESHOLD):
            blink_detected = True
        else:
            blink_detected = False

        # get top left and bottom right coordinates for each eye
        left_eye_top_left = (left_eye[0][0], left_eye[1][1])
        left_eye_bottom_right = (left_eye_top_left[0] + int(left_eye_x1), left_eye_top_left[1] + int(left_eye_y1))
        right_eye_top_left = (right_eye[0][0], right_eye[1][1])
        right_eye_bottom_right = (right_eye_top_left[0] + int(right_eye_x1), right_eye_top_left[1] + int(right_eye_y1))

        # crop each eye out of the full sized image with the face
        left_eye_img = histequ[left_eye_top_left[1]:left_eye_bottom_right[1], left_eye_top_left[0]:left_eye_bottom_right[0]]
        right_eye_img = histequ[right_eye_top_left[1]:right_eye_bottom_right[1], right_eye_top_left[0]:right_eye_bottom_right[0]]

        # enlarge the extracted eye images by 50x for easier viewing
        left_eye_img = cv2.resize(left_eye_img, (50 * left_eye_img.shape[1], 50 * left_eye_img.shape[0]))
        right_eye_img = cv2.resize(right_eye_img, (50 * right_eye_img.shape[1], 50 * right_eye_img.shape[0]))

        # draw rectangles around the left and right eyes in the original feed frame
        cv2.rectangle(img, left_eye_top_left, left_eye_bottom_right, (255, 0, 0), 1)
        cv2.rectangle(img, right_eye_top_left, right_eye_bottom_right, (0, 255, 0), 1)

        # apply binary invert threshold in prep for blob detection
        _, left_eye_img_thresh = cv2.threshold(left_eye_img, EYE_BININV_MIN_THRESHOLD, EYE_BININV_MAX_THRESHOLD, cv2.THRESH_BINARY_INV)
        _, right_eye_img_thresh = cv2.threshold(right_eye_img, EYE_BININV_MIN_THRESHOLD, EYE_BININV_MAX_THRESHOLD, cv2.THRESH_BINARY_INV)
        
        # detect eye blob keypoints
        left_eye_kp = sbd_detector.detect(left_eye_img_thresh)
        right_eye_kp = sbd_detector.detect(right_eye_img_thresh)

        # draw keypoints to visualise
        left_eye_img = cv2.drawKeypoints(left_eye_img, left_eye_kp, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        right_eye_img = cv2.drawKeypoints(right_eye_img, right_eye_kp, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display the cropped left eye image
        cv2.imshow("(f) Left Eye Detection", left_eye_img)
        cv2.imshow("(f) Right Eye Detection", right_eye_img)


    cv2.imshow("(e) Face Detection", img)

    # calculate tiredness score (rolling average of EAR)
    tiredness_score = np.average(frame_EAR_log)

    # alert if tiredness score is below threshold
    if tiredness_score < TIREDNESS_EAR_THRESHOLD:
        tiredness_alert = True
    else:
        tiredness_alert = False

    # record the end time of frame processing iteration
    frame_processing_end = time.time()

    # calculate and append frame processing time to array
    frame_processing_times.append(frame_processing_end - frame_processing_start)

    print(len(frame_processing_times), frame_processing_end - frame_processing_start)

    with console:
        num_frames = len(frame_processing_times)
        sum_fpt = np.sum(frame_processing_times)
        if EAR_alert:
            tiredness_alert = "TIREDNESS ALERT!"
        else:
            tiredness_alert = "---"
        if num_frames > 0:
            avg_fpt = sum_fpt / num_frames
            console.print(
                format.format(
                    num_frames,
                    round(avg_fpt * 1000, 2),
                    target_fpt,
                    math.trunc(1/avg_fpt),
                    capture_fps,
                    blink_detected,
                    round(avg_EAR, 2),
                    round(tiredness_score, 2),
                    tiredness_alert
                )
            )

    # Pause at start to give enough time to move windows to good locations
    if paused:
        while True:
            key = cv2.waitKey(1)
            # Resume feed with 's' key
            if key == ord('s'):
                paused = False
                break
    
    # Capture the next frame of the video feed
    success, img = capture.read()
