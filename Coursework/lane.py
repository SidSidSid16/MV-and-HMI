import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from consoledraw import Console

# Frame capture parameters
VIDEO_CAPTURE = "/Users/sid/Desktop/another_test.mp4"
# VIDEO_CAPTURE = "/Users/sid/Desktop/Local_cropped.mp4"
# VIDEO_CAPTURE = "/Users/sid/Desktop/example_feed_[8].mp4"
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
# Frame crop parameters
CROP_MIN_HEIGHT = 340
CROP_MAX_HEIGHT = 480
CROP_MIN_WIDTH = 100
CROP_MAX_WIDTH = 560
# Canny parameters
CANNY_THRESHOLD_SIGMA = 0.4
# Canny output crop parameters
RECT_0_0 = (165, 0)
RECT_1_0 = (280, 0)
RECT_0_1 = (0, 140)
RECT_1_1 = (460, 140)
# The Hough Transform Parameters
HT_RHO = 1                      # The resolution of the parameter r in pixels. default is 1 pixel. (OpenCV docs)
HT_THETA = np.pi / 180          # The resolution of the parameter θ in radians. default is 1 degree (CV_PI/180). (OpenCV docs)
HT_THRESHOLD = 60               # The minimum number of intersections to "detect" a line. (OpenCV docs)
HT_MIN_LINE_ANGLE = 130         # Minimum angle for line to be detected
HT_MAX_LINE_ANGLE = 140         # Maximum angle for line to be detected
# Lane departure parameters
LEFT_ALERT_THRESHOLD = 100
RIGHT_ALERT_THRESHOLD = 350
LANE_POSITION_VARIANCE_THRESHOLD = 3
MIN_LANE_POSITION_RECORDINGS = 500
MAX_LANE_POSITION_RECORDINGS = 2000
LPV_ALERT_THRESHOLD = 1500
# Program output parameters
USE_CONSOLEDRAW = True          # If true, paramters are output in a user-friendly way. If false, real-time metrics
                                # are output for each frame for efficient tranferring into MS Excel for processing

# IMPORTANT: when running this Python script with ConsoleDraw output enabled, it is imperative that the console windows
#            that's used to start this script is big enough for the output, the program will fail to start with an error:
#            "ValueError: The console is too small to display the buffer."

# instantiate console from consoledraw package
if USE_CONSOLEDRAW:
    console = Console()
    format = """
        Tot. frames: {}
        Avg. FPT: {} ms     (target: {} ms)
        Avg. FPS: {} fps    (target: {} fps)
        Lane Departure:
            L: {}
            V: {}
            R: {}
            LPV: {}       (# readings: {})
            {}
            {}
    """
else:
    print("Frame# FPT LPV LanPosItems#")

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
cv2.namedWindow("(b) Cropped")
cv2.namedWindow("(c) Grayscale")
cv2.namedWindow("(d) Histogram Equalisation")
cv2.namedWindow("(e) Gaussian Blur")
cv2.namedWindow("(f) Canny")
cv2.namedWindow("(g) Clean Canny")
cv2.namedWindow("(h) Hough Transform")

# initialise the array to store frame processing times
frame_processing_times = []

# get target FPS and FPT from input stream
capture_fps = round(capture.get(cv2.CAP_PROP_FPS), 0)
target_fpt = round((1/capture_fps) * 1000, 2)

# initialise lane assist data
right_lane_alert = False
left_lane_alert = False
lane_positions = []
car_positions = []
lane_position_variance = 0

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

    # pause and enter measure mode when 'm' is pressed
    if key == ord('m'):
        # initialise matplotlib sub plots
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        # load in subplots
        ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin='upper')
        ax[0,1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), origin='upper')
        ax[0,2].imshow(grayscale, cmap='gray', origin='upper')
        ax[1,0].imshow(histequ, cmap='gray', origin='upper')
        ax[1,1].imshow(gaussian, cmap='gray', origin='upper')
        ax[1,2].imshow(canny, cmap='gray', origin='upper')
        ax[2,0].imshow(clean_canny, cmap='gray', origin='upper')
        # turn off axis for the columns that is spanned
        ax[2,1].axis('off')
        ax[2,2].axis('off')
        # The Hough plot to span two columns
        ax[2,1] = plt.subplot2grid((3, 3), (2, 1), colspan=2)
        # load in the Hough output as a subplot
        ax[2,1].imshow(cv2.cvtColor(cropped_lines, cv2.COLOR_BGR2RGB), origin='upper')
        # set subplot titles
        ax[0,0].set_title("(a) Original")
        ax[0,1].set_title("(b) Cropped")
        ax[0,2].set_title("(c) Grayscale")
        ax[1,0].set_title("(d) Histogram Equalisation")
        ax[1,1].set_title("(e) Gaussian Blur")
        ax[1,2].set_title("(f) Canny")
        ax[2,0].set_title("(g) Clean Canny")
        ax[2,1].set_title("(h) Hough Transform")
        plt.draw()
        plt.show()

    # enter paused mode with 'p' key
    if key == ord('p'):
        paused = True

    # record the start time of frame processing iteration
    frame_processing_start = time.time()

    # Display original video feed frame in window
    cv2.imshow("(a) Original", img)
    
    # Crop the image to isolate ROI
    cropped = img[CROP_MIN_HEIGHT:CROP_MAX_HEIGHT, CROP_MIN_WIDTH:CROP_MAX_WIDTH]
    # Display cropped video feed frame in window
    cv2.imshow("(b) Cropped", cropped)
    
    # Single-channel converstion using grayscale filter
    grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # Display cropped video feed frame in window
    cv2.imshow("(c) Grayscale", grayscale)
    
    # Histogram equalisation to improve image contrast
    histequ = cv2.equalizeHist(grayscale)
    # Display histogram equalised frame in window
    cv2.imshow("(d) Histogram Equalisation", histequ)
    
    # Gaussian Blur to reduce noise and smoothening
    gaussian = cv2.GaussianBlur(histequ,(5,5),0)
    # Display blurred output frame in window
    cv2.imshow("(e) Gaussian Blur", gaussian)
    
    # Compute the median single-channel pixel intensities
    gaussian_median = np.median(gaussian)
    # Compute threshold values using image median and constant sigma offset
    lower_threshold = int(max(0, (1.0 - CANNY_THRESHOLD_SIGMA) * gaussian_median))
    upper_threshold = int(min(255, (1.0 + CANNY_THRESHOLD_SIGMA) * gaussian_median))
    # Perform the Canny edge detection
    canny = cv2.Canny(gaussian, lower_threshold, upper_threshold)
    # Display all Canny-detected edges
    cv2.imshow("(f) Canny", canny)

    # Remove Canny detections outside of the region of interest
    mask = np.zeros_like(canny)
    # using Numpy, a mask can be created to eliminate Canny data outside of POLY
    vertices = np.array([[RECT_0_0, RECT_1_0, RECT_1_1, RECT_0_1]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    # apply bitmask to eliminate data outside of ROI
    clean_canny = cv2.bitwise_and(canny, mask)
    cv2.imshow("(g) Clean Canny", clean_canny)

    # The Standard Hough Line Transform
    lines = cv2.HoughLines(clean_canny, HT_RHO, HT_THETA, HT_THRESHOLD, None, 0, 0)
    # Copy images that will display the results in BGR
    cropped_lines = cropped.copy()
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            # scale lines to fit frame
            frame_width = CROP_MAX_WIDTH - CROP_MIN_WIDTH
            frame_height = CROP_MAX_HEIGHT - CROP_MIN_HEIGHT
            # ensure that pt1 and pt2 fit within the frame using scaling and tranforming
            if pt1[0] < 0 or pt1[0] > frame_width or pt1[1] < 0 or pt1[1] > frame_height or pt2[0] < 0 or pt2[0] > frame_width or pt2[1] < 0 or pt2[1] > frame_height:
                # Calculate line equation y = mx + c
                m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]) if pt2[0] - pt1[0] != 0 else 1
                c = pt1[1] - m * pt1[0]
                # Clip points to fit within the frame
                if pt1[0] < 0 or pt1[0] > frame_width or pt1[1] < 0 or pt1[1] > frame_height:
                    if pt1[0] < 0:
                        pt1 = (0, int(m * 0 + c))
                    elif pt1[0] > frame_width:
                        pt1 = (frame_width, int(m * frame_width + c))
                    if pt1[1] < 0:
                        pt1 = (int(-c / m), 0)
                    elif pt1[1] > frame_height:
                        pt1 = (int((frame_height - c) / m), frame_height)
                if pt2[0] < 0 or pt2[0] > frame_width or pt2[1] < 0 or pt2[1] > frame_height:
                    if pt2[0] < 0:
                        pt2 = (0, int(m * 0 + c))
                    elif pt2[0] > frame_width:
                        pt2 = (frame_width, int(m * frame_width + c))
                    if pt2[1] < 0:
                        pt2 = (int(-c / m), 0)
                    elif pt2[1] > frame_height:
                        pt2 = (int((frame_height - c) / m), frame_height)
            # compute the angle of the detected road lane marking
            line_angle = np.degrees(np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0]))
            # only process the lines that are within the specified angle range to filter false detections
            if (HT_MIN_LINE_ANGLE < abs(line_angle) < HT_MAX_LINE_ANGLE):
                if (pt1[1] == frame_height):
                    # this is the detection of the left lane marking
                    lane_positions.append((pt1[0], -1))
                    if (pt1[0] > LEFT_ALERT_THRESHOLD):
                        # alert if position is past the threshold
                        left_lane_alert = True
                        line_colour = (0,0,255)
                    else:
                        # deactivate alert otherwise
                        left_lane_alert = False
                        line_colour = (0,255,0)
                    # draw the circle to mark the base position of the lane marking
                    cv2.circle(cropped_lines, center=pt1, radius=10, thickness=2, color=line_colour)
                else:
                    # this is the detection of the right lane marking
                    lane_positions.append((-1, pt2[0]))
                    if (pt2[0] < RIGHT_ALERT_THRESHOLD):
                        # alert if position is past the threshold
                        right_lane_alert = True
                        line_colour = (0,0,255)
                    else:
                        # deactivate alert otherwise
                        right_lane_alert = False
                        line_colour = (0,255,0)
                    # draw the circle to mark the base position of the lane marking
                    cv2.circle(cropped_lines, center=pt2, radius=10, thickness=2, color=line_colour)
                # display the line
                cv2.line(cropped_lines, pt1, pt2, line_colour, 2)

    cv2.imshow("(h) Hough Transform", cropped_lines)

    # check if there are any recorded lane position
    if (len(lane_positions) > MIN_LANE_POSITION_RECORDINGS):
        # retrieve left and right lane marking positions
        left_positions = list(zip(*lane_positions))[0]
        right_positions = list(zip(*lane_positions))[1]
        # convert to numpy arrays for faster computation
        left_positions_arr = np.array(left_positions)
        right_positions_arr = np.array(right_positions)
        # find indices of valid numbers (numbers not equal to -1)
        left_valid_indices = np.where(left_positions_arr != -1)[0]
        right_valid_indices = np.where(right_positions_arr != -1)[0]
        # find indices where -1 (invalid values) needs replacing with interpolated values
        left_replace_indices = np.where(left_positions_arr == -1)[0]
        right_replace_indices = np.where(right_positions_arr == -1)[0]
        # calculate the interpolated values
        left_interpolated_values = np.interp(left_replace_indices, left_valid_indices, left_positions_arr[left_valid_indices])
        right_interpolated_values = np.interp(right_replace_indices, right_valid_indices, right_positions_arr[right_valid_indices])
        # replace -1 with interpolated values
        left_positions_arr[left_replace_indices] = left_interpolated_values
        right_positions_arr[right_replace_indices] = right_interpolated_values
        # update non-numpy array
        lane_positions = list(zip(left_positions_arr.tolist(), right_positions_arr.tolist()))
        # calculate the car's position w.r.t the lanes
        car_positions.append(
            (lane_positions[-1][0] + lane_positions[-1][1])/2
        )
        # calculate variance of car's position w.r.t the lanes
        lane_position_variance = np.var(car_positions)
        # clear older lane position readings (garbage-collection of older recordings)
        if (len(lane_positions) > MAX_LANE_POSITION_RECORDINGS):
            # get number of excess
            excess_vals = len(lane_positions) - MAX_LANE_POSITION_RECORDINGS
            # clear the excess from the start of the list (clearing older excess values)
            del lane_positions[:excess_vals]

    # record the end time of frame processing iteration
    frame_processing_end = time.time()

    # calculate and append frame processing time to array
    frame_processing_times.append(frame_processing_end - frame_processing_start)

    if USE_CONSOLEDRAW:
        with console:
            num_frames = len(frame_processing_times)
            sum_fpt = np.sum(frame_processing_times)
            try:
                current_lane_pos = lane_positions[-1]
                current_car_pos = car_positions[-1]
                if left_lane_alert:
                    lane_alert = "LEFT LANE DEPARTURE ALERT!"
                elif right_lane_alert:
                    lane_alert = "RIGHT LANE DEPARTURE ALERT!"
                else:
                    lane_alert = "---"
                if lane_position_variance > LPV_ALERT_THRESHOLD:
                    lpv_alert = "LANE POSITION VARIANCE ALERT!"
                else:
                    lpv_alert = "---"
            except:
                current_lane_pos = "---"
                current_car_pos = "---"
                lane_alert = "---"
                lpv_alert = "---"
            if num_frames > 0:
                avg_fpt = sum_fpt / num_frames
                console.print(
                    format.format(
                        num_frames,
                        round(avg_fpt * 1000, 2),
                        target_fpt,
                        math.trunc(1/avg_fpt),
                        capture_fps,
                        current_lane_pos[0],
                        current_car_pos,
                        current_lane_pos[1],
                        math.trunc(lane_position_variance),
                        len(lane_positions),
                        lane_alert,
                        lpv_alert
                    )
                )
    else:
        # print a space-delimited output of real-time metrics for easy copy-pasting into MS Excel for graphing
        print(len(frame_processing_times), frame_processing_end - frame_processing_start, lane_position_variance, len(lane_positions))

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

# destroy windows and release video capture for a clean exit
cv2.destroyAllWindows()
capture.release()
