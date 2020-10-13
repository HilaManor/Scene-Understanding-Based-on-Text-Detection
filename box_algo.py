import numpy as np
import cv2
from scipy.stats import norm
from scipy.signal import find_peaks
from shapely.geometry import Polygon
from charnet.modeling.postprocessing import WordInstance
import re

ST_SUFFIX = '(ST|WAY|STREET|AV|AVE|AVENUE|BD|BV|BVD|BOULEVARD|RD|ROAD)'
STREET_PATERN = r'.*\s' + ST_SUFFIX + '([^a-z]|$)'

print('[+] Loading street names...')
with open(r'.\Data\StreetNamesVocab.txt', 'r') as streets_f:
    street_names = [street.upper().strip('\r\n') for street in streets_f.readlines()]

class BoxInstance:
    def __init__(self, word_instance, color_stats, geometric, is_in_streets_list, mask):
        self.word = word_instance
        self.color_stats = color_stats
        self.geometric = geometric
        self.is_in_streets_list = is_in_streets_list
        self.grade = 0
        self.mask = mask

    def update_grade(self):
        # p_count, dist_nearest_p = _ColorStats.get_goodness_of_fit(self)
        check_key_street_words = bool(re.match(STREET_PATERN, self.word.text, re.IGNORECASE))
            updated_grade = 0
        # if p_count != 1:
        #     updated_grade = 0
        else:
            # streets_sign peak is hue = 74, moves to 41 after the convert
            normalized_hue_mean = np.round((100 * self.color_stats.hue_mean) / 180)
            hue_difference = np.abs(normalized_hue_mean - 41)
            updated_grade = check_key_street_words * 10 +\
                            0.9 * self.is_in_streets_list * (100 - hue_difference)
        self.grade = updated_grade


class _ColorStats:
    def __init__(self, hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std,
                 hues, sats, vals, hist):
        self.hue_mean = hue_mean
        self.hue_std = hue_std
        self.sat_mean = sat_mean
        self.sat_std = sat_std
        self.val_mean = val_mean
        self.val_std = val_std
        self.color_data = (hues, sats, vals)
        self.hist = hist

        hue_hist, bins = np.histogram(hues, bins=int(np.round(np.sqrt(len(hues)))))
        hue_hist_padded = np.hstack([hue_hist[1],hue_hist,hue_hist[-2]])
        peaks, props = find_peaks(hue_hist_padded, prominence=1)
        if len(peaks) > 3:  # todo: check
            new_promin = (np.max(props["prominences"]) - np.min(props["prominences"])) / 2
            # new_promin = np.average(props["prominences"])
            peaks, props = find_peaks(hue_hist_padded, prominence=new_promin)
        peaks_locs = [np.average([bins[bin_i+1 -1],bins[bin_i -1]]) for bin_i in peaks]
        dists_from_peaks = [abs(peak-hue_mean) for peak in peaks_locs]
        self.goodness_of_gauss_fit = {"peaks_count" : len(peaks),
                                      "dist_from_nearest": np.min(dists_from_peaks)
                                                            if len(dists_from_peaks) else None}

    # def get_goodness_of_fit(self):
    #         p_count = self.goodness_of_gauss_fit['peaks_count']
    #         dist_nearest_p = self.goodness_of_gauss_fit['dist_from_nearest']
    #         return p_count, dist_nearest_p

    @staticmethod
    def extract_color_stats(panorama, mask):
        hsv_img = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
        hue_img = hsv_img[:, :, 0]
        saturation_img = hsv_img[:, :, 1]
        value_img = hsv_img[:, :, 2]

#        hue_mean, hue_std = curve_fit(lambda x, mu, sig: norm.pdf(x, loc=mu, scale=sig)).fit(hue_img[mask])
        hue_mean, hue_std = norm.fit(hue_img[mask])
        sat_mean, sat_std = norm.fit(saturation_img[mask])
        val_mean, val_std = norm.fit(value_img[mask])
        hist = cv2.calcHist([hsv_img], [0, 1], mask.astype(np.uint8), [50, 60], [0, 180, 0, 256])
        return _ColorStats(hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std,
                           hue_img[mask], saturation_img[mask], value_img[mask], hist)

    @staticmethod
    def combine_color_stats(color_stats_f, color_stats_l, mask, panorama):
        hues_f, sats_f, vals_f, = color_stats_f.color_data
        hues_l, sats_l, vals_l, = color_stats_l.color_data

        hues = np.hstack((hues_f, hues_l))
        sats = np.hstack((sats_f, sats_l))
        vals = np.hstack((vals_f, vals_l))

        hue_mean, hue_std = norm.fit(hues)
        sat_mean, sat_std = norm.fit(sats)
        val_mean, val_std = norm.fit(vals)
        hsv_img = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0, 1], mask.astype(np.uint8),
                            [50, 60], [0, 180, 0, 256])
        return _ColorStats(hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std,
                           hues, sats, vals, hist)


class _Geometrics:
    def __init__(self, bbox_polygon, horiz_angle, vert_angle, horiz_len, vert_len):
        self.polygon = bbox_polygon
        self.horiz_angle = horiz_angle
        self.vert_angle = vert_angle
        self.horiz_len = horiz_len
        self.vert_len = vert_len

    @staticmethod
    def get_polygon_geometric_properties(polygon):
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
        vert_len = np.sqrt(((point_vert[0] - point_a[0]) ** 2) +
                           ((point_vert[1] - point_a[1]) ** 2))
        return _Geometrics(polygon, horiz_angle, vert_angle, horiz_len, vert_len)

    @staticmethod
    def combine_geometric_properties(geometric_f, geometric_l):
        new_poly = geometric_f.polygon.union(geometric_l.polygon).minimum_rotated_rectangle
        base_poly_comb = _Geometrics.get_polygon_geometric_properties(new_poly)
        base_poly_comb.horiz_angle = np.average((geometric_f.horiz_angle,
                                                 geometric_l.horiz_angle))  # uh..
        base_poly_comb.vert_angle = np.average((geometric_f.vert_angle,
                                                 geometric_l.vert_angle))  # uh..
        return base_poly_comb


def __search_in_street_names(text):
    is_street_pattern = re.match(STREET_PATERN, text, re.IGNORECASE)

    # if text in EXCLUSIONS:
    #     return False
    if is_street_pattern:
        return text in street_names or is_street_pattern.group(1) in street_names
    else:
        return text in street_names or (text + " street") in street_names


def compund_bboxes(first_bbox, last_bbox, panorama, amprecent=False):
    new_bboxes = first_bbox.geometric.polygon.union(
        last_bbox.geometric.polygon).minimum_rotated_rectangle.exterior.coords[:-1]
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
    mask = first_bbox.mask | last_bbox.mask
    color_stats = _ColorStats.combine_color_stats(first_bbox.color_stats, last_bbox.color_stats,
                                                  mask, panorama)
    geometric = _Geometrics.combine_geometric_properties(first_bbox.geometric,
                                                          last_bbox.geometric)
    is_in_streets_list = __search_in_street_names(new_text)

    # create the bbox instance
    return BoxInstance(new_word, color_stats, geometric, is_in_streets_list, mask)


def expand_word_data(twords, panorama):
    boxes = []
    print('[+] Gathering words data...')
    for word in twords:
        mask = np.zeros((panorama.shape[0], panorama.shape[1]), dtype=np.uint8)
        poly = np.array(word.word_bbox, dtype=np.int32).reshape((1, 4, 2))
        cv2.fillPoly(mask, poly, 255)
        mask = mask.astype(np.bool)
        color_stats = _ColorStats.extract_color_stats(panorama, mask)
        is_in_streets_list = __search_in_street_names(word.text)
        bbox_polygon = Polygon([(word.word_bbox[0], word.word_bbox[1]),
                                (word.word_bbox[2], word.word_bbox[3]),
                                (word.word_bbox[4], word.word_bbox[5]),
                                (word.word_bbox[6], word.word_bbox[7])])
        geometric = _Geometrics.get_polygon_geometric_properties(bbox_polygon)

        # create the bbox instance
        box = BoxInstance(word, color_stats, geometric, is_in_streets_list, mask)
        boxes.append(box)
    return boxes
