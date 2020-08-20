import numpy as np
import cv2
from scipy.stats import norm


class BoxInstance:
    class ColorStats:
        def __init__(self, hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std):
            self.hue_mean = hue_mean
            self.hue_std = hue_std
            self.sat_mean = sat_mean
            self.sat_std = sat_std
            self.val_mean = val_mean
            self.val_std = val_std

    def __init__(self, word_instance, hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std,
                 is_in_streets_list=False):
        self.word = word_instance
        self.color_stats = BoxInstance.ColorStats(hue_mean, hue_std, sat_mean,
                                                  sat_std, val_mean, val_std)
        self.is_in_streets_list = is_in_streets_list
        self.grade = 0


def expand_word_data(twords, panorama):
    print('[+] Loading street names...')
    with open(r'.\Data\StreetNamesVocab.txt', 'r') as streets_f:
        street_names = [street.upper().strip('\r\n') for street in streets_f.readlines()]
    boxes = []

    print('[+] Gathering words data...')
    for word in twords:
        hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std = __extract_color_stats(panorama,
                                                                                        word)
        is_in_streets_list = __check_in_street_list(word.text, street_names)
        box = BoxInstance(word, hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std,
                          is_in_streets_list)
        boxes.append(box)
    return boxes


def __extract_color_stats(panorama, word):
    mask = np.zeros((panorama.shape[0], panorama.shape[1]), dtype=np.uint8)
    poly = np.array(word.word_bbox, dtype=np.int32).reshape((1, 4, 2))
    cv2.fillPoly(mask, poly, 255)
    mask = mask.astype(np.bool)
    hsv_img = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
    hue_img = hsv_img[:, :, 0]
    saturation_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]
    hue_mean, hue_std = norm.fit(hue_img[mask])
    sat_mean, sat_std = norm.fit(saturation_img[mask])
    val_mean, val_std = norm.fit(value_img[mask])
    return hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std


def __check_in_street_list(word, street_names):
    return word in street_names