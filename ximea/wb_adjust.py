import numpy as np
import cv2
from ximea import xiapi


# Connect to camera
cam = xiapi.Camera()
cam.open_device_by_SN('xxxxxx')  # Enter serial number of your Ximea camera

# Configuration
cam.set_exposure(2000000)
cam.set_gain(0)
cam.set_gammaY(1/2.2)
cam.set_imgdataformat('XI_RGB24')

# Allocate memory for ximea image
img = xiapi.Image()

# Start acquisition
cam.start_acquisition()
cam.get_image(img)
cvimg = img.get_image_data_numpy()

camwinname = 'camera'
cv2.namedWindow(camwinname, cv2.WINDOW_NORMAL)
cv2.resizeWindow(camwinname, (int(cvimg.shape[1]/2), int(cvimg.shape[0]/2)))

cam.set_wb_kb(2.6510791778564453)
cam.set_wb_kg(1.0)
cam.set_wb_kr(1.6091703176498413)

# Preview output from camera
key = -1
while key == -1:
    # cam.set_manual_wb(1)

    cam.get_image(img)
    cvimg = img.get_image_data_numpy()
    cv2.imshow('camera', cvimg)

    # kb = cam.get_wb_kb()
    # kg = cam.get_wb_kg()
    # kr = cam.get_wb_kr()
    # print(kb, kg, kr)

    key = cv2.waitKey(1)

cam.get_image(img)
cvimg = img.get_image_data_numpy()
cv2.imshow('camera', cvimg)
cv2.imwrite('cap.png', cvimg)

# Terminate
cam.stop_acquisition()
cam.close_device()
