# App for interactive camera calibration

import cv2
import numpy as np


CALIBFLAG = 0  # cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST


class MyCamera():  # Modify this class according to your environment
    cap = None

    def initialize(self):
        self.cap = cv2.VideoCapture(0)

    def getGrayscaleFrame(self):
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            return None

    def terminate(self):
        self.cap.release()


def main():
    print("----- Camera calibration -----")
    print()

    print("-- Mode --")
    print("1 : Chessboard pattern")
    print("2 : Dot grid pattern")
    mode = 1
    while True:
        print("Select pattern to use (1 or 2) :")
        s = input()
        s.strip()
        if s[0] == '1':
            mode = 1
            print("Chessboard pattern is selected")
            break
        elif s[0] == '2':
            mode = 2
            print("Dot grid pattern is selected")
            break
    print()

    print("-- Grid size --")
    print("Input grid height :")
    gridHeight = int(input())
    print("Input grid width :")
    gridWidth = int(input())
    print()

    print("-- Capturing --")
    print("Initializing camera ...")
    cam = MyCamera()
    cam.initialize()

    cv2.namedWindow('capturing', cv2.WINDOW_NORMAL)
    cv2.namedWindow('detected grid', cv2.WINDOW_NORMAL)

    print("Commands (enter key while focusing on OpenCV's windows)")
    print("  <space> : Try to find the grid pattern")
    print("  c       : Cancel previous shot (press if you find detection failure)")
    print("  g       : Go to calibration step")
    print("  q       : Quit without calibration")
    print()

    grid3d = np.zeros((gridHeight*gridWidth, 3), np.float32)
    grid3d[:, :2] = np.mgrid[0:gridHeight, 0:gridWidth].T.reshape(-1, 2)

    grid3dList = []
    grid2dList = []
    cancelable = False
    imgShape = None
    while(True):
        frame = cam.getGrayscaleFrame()
        if frame is None:
            print("Failed to get frame.")
            continue

        cv2.imshow('capturing', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cam.terminate()
            return
        elif key == ord('g'):
            break
        elif key == ord(' '):
            imgShape = frame.shape
            res = None
            if mode == 1:
                res = cv2.findChessboardCorners(
                    frame, (gridHeight, gridWidth), None)
            elif mode == 2:
                res = cv2.findCirclesGrid(
                    frame, (gridHeight, gridWidth), None,
                    cv2.CALIB_CB_CLUSTERING | cv2.CALIB_CB_SYMMETRIC_GRID)
            ret, points = res
            if ret:
                cv2.drawChessboardCorners(
                    frame, (gridHeight, gridWidth), points, True)
                cv2.imshow('detected grid', frame)
                grid3dList.append(grid3d)
                grid2dList.append(points)
                cancelable = True
                print("Found (" + str(len(grid2dList)) + " frames)")
            else:
                print("Not found")
        elif key == ord('c') and cancelable:
            grid3dList.pop()
            grid2dList.pop()
            cancelable = len(grid2dList) > 0
            print("Canceled (" + str(len(grid2dList)) + " frames)")
            cv2.destroyWindow('detected grid')
            cv2.namedWindow('detected grid', cv2.WINDOW_NORMAL)
    print()

    print("-- Calibration --")
    ret, camInt, camDist, rvecs, tvecs = cv2.calibrateCamera(
        grid3dList, grid2dList, imgShape, None, None, None, None, CALIBFLAG
    )
    print(str(len(grid2dList)) + " frames are used.")
    print('Image size :', imgShape)
    print('RMS :', ret)
    print('Intrinsic parameters :')
    print(camInt)
    print('Distortion parameters :')
    print(camDist)
    print()

    print("./calibration_result.xml was saved !!")
    print()

    fs = cv2.FileStorage('calibration_result.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('imgShape', imgShape)
    fs.write('rms', ret)
    fs.write('intrinsic', camInt)
    fs.write('distortion', camDist)
    fs.release()


if __name__ == '__main__':
    main()
