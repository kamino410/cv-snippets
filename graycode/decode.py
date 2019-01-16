#coding: UTF-8

import sys
import os
import os.path
import glob
import re
import cv2
import numpy as np


TARGETDIR = './graycode_pattern'
CAPTUREDDIR = './captured'

BLACKTHR = 5
WHITETHR = 40

def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(
            "Usage : python3 calibrate.py <projector image height> <projector image width> [graycode step(default=1)]")
        print()
        return

    step = int(sys.argv[3]) if len(sys.argv) == 4 else 1
    proj_width = int(sys.argv[1])
    proj_height = int(sys.argv[2])
    gc_width = int((proj_width-1)/step)+1
    gc_height = int((proj_height-1)/step)+1

    graycode = cv2.structured_light_GrayCodePattern.create(gc_height, gc_width)
    graycode.setBlackThreshold(BLACKTHR)
    graycode.setWhiteThreshold(WHITETHR)

    re_num = re.compile(r'(\d+)')
    def numerical_sort(text):
        return int(re_num.split(text)[-2])
    filenames = sorted(glob.glob(CAPTUREDDIR + '/capture_*.png'), key=numerical_sort)

    if len(filenames) != graycode.getNumberOfPatternImages() + 2:
        print('Number of images is not right (right number is ' + str(graycode
            .getNumberOfPatternImages() + 2) + ')')
        return

    imgs = []
    for f in filenames:
        imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
    black = imgs.pop()
    white = imgs.pop()
    cam_height = white.shape[0]
    cam_width = white.shape[1]
    print('camera image size :', white.shape)
    print()

    viz_c2p = np.zeros((cam_height, cam_width, 3), np.uint8)

    c2p_list = [] # [((cam x, y), (proj x, y))]
    for y in range(cam_height):
        for x in range(cam_width):
            if int(white[y, x]) - int(black[y, x]) <= BLACKTHR:
                continue
            err, proj_pix = graycode.getProjPixel(imgs, x, y)
            if not err:
                viz_c2p[y, x, :] = [proj_pix[0], proj_pix[1], 128]
                c2p_list.append(((x, y), proj_pix))

    print('=== Result ===')
    print('Decoded c2p correspondences :', len(c2p_list))
    cv2.imwrite('vizualize_c2p.png', viz_c2p)
    print('Vizualized image : \'./visualize_c2p.png\'')
    with open('result_c2p.csv', 'w') as f:
        f.write('cam_x, cam_y, proj_x, proj_y\n')
        for p in c2p_list:
            f.write(str(p[0][0]) + ', ' + str(p[0][1]) + ', ')
            f.write(str(p[1][0]) + ', ' + str(p[1][1]) + '\n')
    
    print('output : \'./result_c2p\'')
    print()


if __name__ == '__main__':
    main()

