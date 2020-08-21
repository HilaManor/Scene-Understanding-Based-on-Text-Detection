import numpy as np
import cv2
from scipy.stats import norm
from shapely.geometry import Polygon
from charnet.modeling.postprocessing import WordInstance

class BoxInstance:
    class ColorStats:
        def __init__(self, hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std,
                     hues, sats, vals):
            self.hue_mean = hue_mean
            self.hue_std = hue_std
            self.sat_mean = sat_mean
            self.sat_std = sat_std
            self.val_mean = val_mean
            self.val_std = val_std
            self.color_data = (hues, sats, vals)

    class Geometrics:
        def __init__(self, bbox_polygon, horiz_angle, vert_angle, horiz_len, vert_len):
            self.polygon = bbox_polygon
            self.horiz_angle = horiz_angle
            self.vert_angle = vert_angle
            self.horiz_len = horiz_len
            self.vert_len = vert_len

    def __init__(self, word_instance, color_stats, geometrics, is_in_streets_list):
        self.word = word_instance
        self.color_stats = color_stats
        self.geometrics = geometrics
        self.is_in_streets_list = is_in_streets_list
        self.grade = 0


def expand_word_data(twords, panorama):
    print('[+] Loading street names...')
    with open(r'.\Data\StreetNamesVocab.txt', 'r') as streets_f:
        street_names = [street.upper().strip('\r\n') for street in streets_f.readlines()]
    boxes = []

    print('[+] Gathering words data...')
    for word in twords:
        color_stats = __extract_color_stats(panorama, word)
        is_in_streets_list = __check_in_street_list(word.text, street_names)
        bbox_polygon = Polygon([(word.word_bbox[0], word.word_bbox[1]),
                                (word.word_bbox[2], word.word_bbox[3]),
                                (word.word_bbox[4], word.word_bbox[5]),
                                (word.word_bbox[6], word.word_bbox[7])])
        geometrics = __get_polygon_geometric_properties(bbox_polygon)

        # create the bbox instance
        box = BoxInstance(word, color_stats, geometrics, is_in_streets_list)
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
    return BoxInstance.ColorStats(hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std,
                                  hue_img[mask], saturation_img[mask], value_img[mask])


def __check_in_street_list(word, street_names):
    return word in street_names


def __get_polygon_geometric_properties(polygon):
    # Exterior looks like this:
    # 3 ----- 2
    # |       |
    # 0 ----- 1
    point_a = polygon.exterior.coords[0]
    point_horiz = polygon.exterior.coords[1]
    point_vert = polygon.exterior.coords[3]
    horiz_angle = np.arctan2(point_horiz[1] - point_a[1], point_horiz[0] - point_a[0])
    vert_angle = np.arctan2(point_vert[1] - point_a[1], point_vert[0] - point_a[0])
    horiz_len = np.sqrt(((point_horiz[0] - point_a[0]) ** 2) +
                        ((point_horiz[1] - point_a[1]) ** 2))
    vert_len = np.sqrt(((point_vert[0] - point_a[0]) ** 2) + ((point_vert[1] - point_a[1]) ** 2))

    return BoxInstance.Geometrics(polygon, horiz_angle, vert_angle, horiz_len, vert_len)


def compund_bboxes(first_bbox, last_bbox, amprecent=False):
    new_bboxes = first_bbox.geometrics.polygon.union(
        last_bbox.geometrics.polygon).minimum_rotated_rectangle.exterior.coords[:-1]
    new_bboxes = np.round(np.array(sum(new_bboxes, ()), dtype=np.float32))

    new_char_scores = np.vstack((first_bbox.word.char_scores,
                                 last_bbox.word.char_scores))
    new_text = first_bbox.word.text + (" & " if amprecent else " ") + last_bbox.word.text

    new_word = WordInstance(
        new_bboxes,
        np.average([first_bbox.word.word_bbox_score, last_bbox.word.word_bbox_score]),
        new_text,
        np.average([first_bbox.word.text_score, last_bbox.word.text_score]),
        new_char_scores
    )

    color_stats = __combine_color_stats(first_bbox.color_stats, last_bbox.color_stats)
    geometrics = __combine_geometric_properties(first_bbox.geometrics, last_bbox.geometrics)
    is_in_streets_list = first_bbox.is_in_streets_list or last_bbox.is_in_streets_list  # todo

    # create the bbox instance
    return BoxInstance(new_word, color_stats, geometrics, is_in_streets_list)


def __combine_color_stats(color_stats_f, color_stats_l):
    hues_f, sats_f, vals_f, = color_stats_f.color_data
    hues_l, sats_l, vals_l, = color_stats_l.color_data

    hues = np.hstack((hues_f, hues_l))
    sats = np.hstack((sats_f, sats_l))
    vals = np.hstack((vals_f, vals_l))

    hue_mean, hue_std = norm.fit(hues)
    sat_mean, sat_std = norm.fit(sats)
    val_mean, val_std = norm.fit(vals)
    return BoxInstance.ColorStats(hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std,
                                  hues, sats, vals)


def __combine_geometric_properties(geometrics_f, geometrics_l):
    new_poly = geometrics_f.polygon.union(geometrics_l.polygon).minimum_rotated_rectangle
    return __get_polygon_geometric_properties(new_poly)
