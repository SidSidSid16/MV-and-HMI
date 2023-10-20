import cv2
import numpy as np

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
number_of_images = 10
imglist = []
success = True

# Loop through the indices of images to be captured
for imgnum in range(number_of_images):
    # Capture images continuously and wait for a key press
    while success and cv2.waitKey(1) == -1:
        # Read an image from the VideoCapture instance
        success, img = cap.read()
        # Display the image
        cv2.imshow("Video Stream", img)
    # When we exit the capture loop we save the last image and repeat
    imglist.append(img)
    cv2.imshow("Captured Image", img)
    print("Image Captured")

# The image index loop ends when number_of_images have been captured
print("Captured", len(imglist), "images")
# Save all images to image files for later use
for imgnum, img in enumerate(imglist):
    cv2.imwrite("Image%03d.png" % (imgnum), img)
# Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")
cap.release()