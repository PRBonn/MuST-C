import cv2
import numpy as np


def get_pinks(img_bgr, hue_thresh=(138, 180)):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    is_pink = np.logical_and(img_hsv[:,:,0] > hue_thresh[0], img_hsv[:,:,0] < hue_thresh[1])
    return is_pink


def get_exR(image_clean):
    image = cv2.GaussianBlur(image_clean, (13, 13), 0)
    img_b = image[:,:,0]
    img_g = image[:,:,1]
    img_r = image[:,:,2]

    exR = img_r * 2. - img_b - img_g
    return exR
