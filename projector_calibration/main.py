# coding : UTF-8
import numpy as np
import cv2
import csv
import sys


def load_files(file1, file2):
    img_points = []
    with open(file1) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                img_points.append(
                    np.array([int(row[0]), int(row[1])], np.float32))
    obj_points = []
    with open(file2) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                obj_points.append(
                    np.array([float(row[0]), float(row[1]), float(row[2])], np.float32))
    return (np.array(img_points), np.array(obj_points))


def main():
    if len(sys.argv) != 5:
        print(
            'python3 calibrate_projector.py [image_points.csv] [object_points.csv] [image width] [image height]')

    img_points, obj_points = load_files(sys.argv[1], sys.argv[2])
    width = int(sys.argv[3])
    height = int(sys.argv[4])

    cammat = np.array(
        [[width, 0, width/2], [0, width, height/2], [0, 0, 1]], np.float32)
    dist = np.zeros(5)

    retval, cammat, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj_points], [img_points], (width, height), cammat, dist, None, None,
        cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST)
    print('re-projection error')
    print(retval)
    print('intrinsic parameters')
    print(cammat)
    print('rvecs')
    print(rvecs)
    print('tvecs')
    print(tvecs)


if __name__ == '__main__':
    main()
