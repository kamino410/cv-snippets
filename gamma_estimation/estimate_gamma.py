#coding: UTF-8
import numpy as np
import cv2
from scipy.optimize import least_squares


def main():
    """
    ガンマ値1に設定し、白飛びしない程度にしぼりをかけたカメラで
    輝度値[0, 5, 10, ... , 255]を表示したディスプレイの画像を
    `gamma_[輝度値].png`の名前で保存してから実行してください。
    画像中央部分の25画素の平均値を利用します。
    """
    rads = []
    for p in range(0, 256, 5):
        img = cv2.imread('gamma/gamma_' + str(p) + '.png', 0)
        half_h = int(img.shape[0]/2)
        half_w = int(img.shape[1]/2)
        rad = np.average(img[half_h-2:half_h+2, half_w-2:half_w+2])
        print(rad)
        rads.append(rad)

    def res_func(gamma, xs, ys):
        return ys - np.power(xs, gamma)

    xs = np.linspace(0, 255, 52)/255
    min_val = rads[0]
    max_val = rads[-1]
    ys = np.array([(y - min_val)/(max_val - min_val) for y in rads])

    init_gamma = 1

    res = least_squares(res_func, init_gamma, args=(xs, ys), method='lm')
    print('success:', res.success)
    print('RMS', np.sqrt(np.average(res.fun**2)))
    print('gamma:', res.x)


if __name__ == '__main__':
    main()
