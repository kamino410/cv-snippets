# coding: UTF-8

# Install XIMEA software package
# Copy 'XIMEA\API\Python\v3\ximea' to 'PythonXX\Lib'

from ximea import xiapi
import cv2
import numpy as np

# Connect to camera
cam = xiapi.Camera()
cam.open_device_by_SN('XXXXXXXX')  # Enter serial number of your Ximea camera

# Configuration
cam.set_exposure(100000)
cam.set_gain(1)
cam.set_gammaY(1)

# Allocate memory for ximea image
img = xiapi.Image()

# Start acquisition
cam.start_acquisition()

# Preview output from camera
key = -1
while key == -1:
    cam.get_image(img)
    cvimg = img.get_image_data_numpy()
    cv2.imshow('camera', cvimg)
    key = cv2.waitKey(1)

cv2.namedWindow('pattern', cv2.WINDOW_NORMAL)
cv2.moveWindow('pattern', 1920, 0)
cv2.setWindowProperty(
    'pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

key = -1
while key == -1:
    cam.get_image(img)
    cvimg = img.get_image_data_numpy()
    cv2.imshow('camera', cvimg)
    key = cv2.waitKey(1)

# Preview output from camera
for v in range(0, 256, 5):
    pat = v*np.ones((1080, 1920), np.uint8)
    cv2.imshow('pattern', pat)
    cv2.waitKey(400)

    cam.get_image(img)
    cvimg = img.get_image_data_numpy()
    cv2.imwrite('gamma_' + str(v) + '.png', cvimg)

# Terminate
cam.stop_acquisition()
cam.close_device()
