#coding: UTF-8

import sys
import os
import os.path
import cv2
import numpy as np

TARGETDIR = './graycode_pattern'
CAPTUREDDIR = './captured'


def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(
            "Usage : python3 gen_graycode_imgs.py <projector image height> <projector image width> [graycode step(default=1)]")
        print()
        return

    step = int(sys.argv[3]) if len(sys.argv) == 4 else 1
    width = int(sys.argv[1])
    height = int(sys.argv[2])
    gc_width = int((width-1)/step)+1
    gc_height = int((height-1)/step)+1

    graycode = cv2.structured_light_GrayCodePattern.create(gc_height, gc_width)
    patterns = graycode.generate()[1]

    # expand image size
    exp_patterns = []
    for pat in patterns:
        img = np.zeros((width, height), np.uint8)
        for y in range(height):
            for x in range(width):
                img[x, y] = pat[int(x/step), int(y/step)]
        exp_patterns.append(img)

    exp_patterns.append(255*np.ones((width, height), np.uint8))  # white
    exp_patterns.append(np.zeros((width, height), np.uint8))    # black

    if not os.path.exists(TARGETDIR):
        os.mkdir(TARGETDIR)

    for i, pat in enumerate(exp_patterns):
        cv2.imwrite(TARGETDIR + '/pattern_' + str(i).zfill(2) + '.png', pat)

    print('=== Result ===')
    print('\'' + TARGETDIR + '/pattern_00.png ~ pattern_' +
          str(len(exp_patterns)-1) + '.png \' were generated')
    print()
    print('=== Next step ===')
    print('Project patterns and save captured images as \'' +
          CAPTUREDDIR + '/capture_*.png\'')
    print()


if __name__ == '__main__':
    main()

