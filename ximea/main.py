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

# Terminate
cam.stop_acquisition()
cam.close_device()
