#coding: UTF-8

import sys
import os
import os.path
import glob
import cv2
import numpy as np


CAPTUREDDIR = './captured'
CALIBFLAG = 0  # cv2.CALIB_FIX_K3


def calibFromImages(dirname, chess_shape, chess_block_size):
    if not os.path.exists(dirname):
        print('Directory \'' + dirname + '\' was not found')
        return None

    filenames = sorted(glob.glob(dirname + '/*'))
    if len(filenames) == 0:
        print('No image was found in \'' + dirname + '\'')
        return None

    print('=== Camera Calibration ===')

    objp = np.zeros((chess_shape[0]*chess_shape[1], 3), np.float32)
    objp[:, :2] = chess_block_size * \
        np.mgrid[0:chess_shape[0], 0:chess_shape[1]].T.reshape(-1, 2)

    print('Finding chess corners in input images ...')
    objp_list = []
    imgp_list = []
    img_shape = None
    for f in filenames:
        print('  ' + f + ' : ', end='')
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img_shape is None:
            img_shape = img.shape
        elif img_shape != img.shape:
            print('Mismatch size')
            continue
        ret, imgp = cv2.findChessboardCorners(img, chess_shape, None)
        if ret:
            print('Found')
            objp_list.append(objp)
            imgp_list.append(imgp)
        else:
            print('Not found')

    print(' ', len(objp_list), 'images are used')
    ret, cam_int, cam_dist, rvecs, tvecs = cv2.calibrateCamera(
        objp_list, imgp_list, img_shape, None, None, None, None, CALIBFLAG
    )
    print('Image size :', img_shape)
    print('RMS :', ret)
    print('Intrinsic parameters :')
    print(cam_int)
    print('Distortion parameters :')
    print(cam_dist)
    print()

    rmtxs = list(map(lambda vec: cv2.Rodrigues(vec)[0], rvecs))

    fs = cv2.FileStorage('calibration.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('img_shape', img_shape)
    fs.write('rms', ret)
    fs.write('intrinsic', cam_int)
    fs.write('distortion', cam_dist)
    fs.write('rotation_vectors', np.array(rvecs))
    fs.write('rotation_matrixes', np.array(rmtxs))
    fs.write('translation_vectors', np.array(tvecs))
    fs.release()

    return (img_shape, ret, cam_int, cam_dist, rvecs, tvecs)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        chess_shape = (int(sys.argv[1]), int(sys.argv[2]))
        chess_block_size = float(sys.argv[3])

        calibFromImages(CAPTUREDDIR, chess_shape, chess_block_size)
    else:
        print('Usage :')
        print('  Save captured images into \'' + CAPTUREDDIR + '\'')
        print(
            '  Run \'python3 caliblate_camera_from_images.py <num of chess corners in vert> <num of chess corners in hori> <chess block size(m or mm)>')
