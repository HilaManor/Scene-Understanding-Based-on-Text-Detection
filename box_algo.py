"""This module handles the middle part of the 2nd algorithm stage: Extracting the street-signs

This module holds the functionality of extracting the geometric and color data of every word that
was extracted. When signs need to be combined, the data is combined through this module
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import cv2
from scipy.stats import norm
from scipy.signal import find_peaks
from shapely.geometry import Polygon
from charnet.modeling.postprocessing import WordInstance
import re

# ~~~~~~~~~~~~~~~~~~~~~~~~ Constants ~~~~~~~~~~~~~~~~~~~~~~~
ST_SUFFIX = '(ST|WAY|STREET|AV|AVE|AVENUE|BD|BV|BVD|BOULEVARD|RD|ROAD)'
STREET_PATERN = r'.*\s' + ST_SUFFIX + '([^a-z]|$)'

print('[+] Loading street names...')
with open(r'.\Data\StreetNamesVocab.txt', 'r') as streets_f:
    street_names = [street.upper().strip('\r\n') for street in streets_f.readlines()]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~


class BoxInstance:
    """Wraps a WordInstance class with the corresponding bounding box's color and geometric data

    Public Functions:
        update_grade - Update the street-sign grade of the instance

    Variables:
        word - WordInstance for the bounding box
        color_stats - _ColorStats instance holding the color data
        geometric - _Geometrics instance holding the geometric data
        is_in_streets_list - Boolean for weather the word appears in the street names list
            extracted at init
        grade - possibility of the insatnce to be a street sign
        key_street_word - Boolean for weather the word hold a keyword, such as "st", "ave", etc.
        mask - mask for the panorama to this instance's area
    """
    def __init__(self, word_instance, color_stats, geometric, is_in_streets_list, mask):
        self.word = word_instance
        self.color_stats = color_stats
        self.geometric = geometric
        self.is_in_streets_list = is_in_streets_list
        self.grade = 0
        self.key_street_word = False
        self.mask = mask

    def update_grade(self):
        """Update the street-sign grade of the instance

        :return: None
        """
        check_key_street_words = bool(re.match(STREET_PATERN, self.word.text, re.IGNORECASE))
        self.key_street_word = check_key_street_words
        if not self.color_stats.goodness_of_gauss_fit["peaks_count"]:
            updated_grade = 0
        else:
            # green street signs have usually green(74) background color
            hue_mean_dist = np.round(100 * np.abs(self.color_stats.hue_mean - 74) / 180)
            fit_grade = max(0, self.color_stats.goodness_of_gauss_fit["dist_from_nearest"] - 7)
            updated_grade = 10 * check_key_street_words + \
                            5 * self.is_in_streets_list + \
                            0.85 * ((100 - hue_mean_dist) *
                                    (100 - fit_grade) * 0.01)
        self.grade = updated_grade


class _ColorStats:
    """Color data of an area (bounding box)

        Public Functions:
            extract_color_stats - Find the color data of an area in a panorama
            combine_color_stats - Combine the color data of 2 areas

        Variables:
            hue_mean - the mean of a fitted gaussian curve to the hue channel of the area
            sat_mean - the mean of a fitted gaussian curve to the saturation channel of the area
            val_mean - the mean of a fitted gaussian curve to the value channel of the area
            color_data - all the area's pixels' HSV data
            hist - 2d histogram for hues and saturations in the area
            goodness_of_gauss_fit - how well did the gaussian curve was fitted to the hue channel
                                    described as a dict:
                                        peaks_count: the amount of prominent peaks in the histogram
                                        dist_from_nearest: distance from the mean to its nearest
                                            peak
        """
    def __init__(self, hue_mean, sat_mean, val_mean, hues, sats, vals, hist):
        self.hue_mean = hue_mean
        self.sat_mean = sat_mean
        self.val_mean = val_mean
        self.color_data = (hues, sats, vals)
        self.hist = hist

        hue_hist, bins = np.histogram(hues, bins=int(np.round(np.sqrt(len(hues)))))
        hue_hist_padded = np.hstack([hue_hist[1], hue_hist, hue_hist[-2]])
        peaks, props = find_peaks(hue_hist_padded, prominence=1)
        if len(peaks) > 3:
            new_prominence = (np.max(props["prominences"]) - np.min(props["prominences"])) / 2
            peaks, props = find_peaks(hue_hist_padded, prominence=new_prominence)
        peaks_locs = [np.average([bins[bin_i+1 - 1], bins[bin_i - 1]]) for bin_i in peaks]
        dists_from_peaks = [abs(peak-hue_mean) for peak in peaks_locs]
        self.goodness_of_gauss_fit = {"peaks_count": len(peaks),
                                      "dist_from_nearest": np.min(dists_from_peaks)
                                      if len(dists_from_peaks) else None}

    @staticmethod
    def extract_color_stats(panorama, mask):
        """Find the color data of a given area in the panorama

        :param panorama: the panorama image to extract the color data from
        :param mask: the mask to apply to the panorama
        :return: _ColorStats instance representing the area's color data
        """
        hsv_img = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
        hue_img = hsv_img[:, :, 0]
        saturation_img = hsv_img[:, :, 1]
        value_img = hsv_img[:, :, 2]

        hue_mean, _ = norm.fit(hue_img[mask])
        sat_mean, _ = norm.fit(saturation_img[mask])
        val_mean, _ = norm.fit(value_img[mask])
        hist = cv2.calcHist([hsv_img], [0, 1], mask.astype(np.uint8), [50, 60], [0, 180, 0, 256])
        return _ColorStats(hue_mean, sat_mean, val_mean, hue_img[mask], saturation_img[mask],
                           value_img[mask], hist)

    @staticmethod
    def combine_color_stats(color_stats_f, color_stats_l, mask, panorama):
        """Combine the color data of 2 areas in a panorama

        :param color_stats_f: 1st area's color data to connect (its word is to the left of the 2nd)
        :param color_stats_l: 2nd area's color data to connect
        :param mask: the combined mask of the connecting areas
        :param panorama: panorama of the words, to which the mask will be applied
        :return: new _ColorStats Containing the combined color data
        """
        hues_f, sats_f, vals_f, = color_stats_f.color_data
        hues_l, sats_l, vals_l, = color_stats_l.color_data

        hues = np.hstack((hues_f, hues_l))
        sats = np.hstack((sats_f, sats_l))
        vals = np.hstack((vals_f, vals_l))

        hue_mean, _ = norm.fit(hues)
        sat_mean, _ = norm.fit(sats)
        val_mean, _ = norm.fit(vals)
        hsv_img = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0, 1], mask.astype(np.uint8),
                            [50, 60], [0, 180, 0, 256])
        return _ColorStats(hue_mean, sat_mean, val_mean, hues, sats, vals, hist)


class _Geometrics:
    """Geometric data of a bounding box

    Public Functions:
        get_polygon_geometric_properties - find the geometric properties of a given polygon
        combine_geometric_properties - Combine the geometric properties of 2 polygons

    Variables:
        polygon - the minimum rotated bounding rectangle (the box)
        horiz_angle - angle of the horiz edge of the polygon
        vert_angle - angle of the vertical edge of the polygon
        horiz_len - horizontal length of the polygon
        vert_len - vertical length of the polygon
    """
    def __init__(self, bbox_polygon, horiz_angle, vert_angle, horiz_len, vert_len):
        self.polygon = bbox_polygon
        self.horiz_angle = horiz_angle
        self.vert_angle = vert_angle
        self.horiz_len = horiz_len
        self.vert_len = vert_len

    @staticmethod
    def get_polygon_geometric_properties(polygon):
        """Find the geometric properties of a given polygon

        :param polygon: polygon to find properties for
        :return: _Geometrics instance representing the polygon
        """

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
        """Combine the geometric properties of 2 polygons

        :param geometric_f: First polygon to connect (its word is to the left of the second)
        :param geometric_l: Second polygon to connect
        :return: new polygon (minimum rotated bounding rectangle containing both polygons)
        """
        new_poly = geometric_f.polygon.union(geometric_l.polygon).minimum_rotated_rectangle
        base_poly_comb = _Geometrics.get_polygon_geometric_properties(new_poly)
        base_poly_comb.horiz_angle = np.average((geometric_f.horiz_angle,
                                                 geometric_l.horiz_angle))
        base_poly_comb.vert_angle = np.average((geometric_f.vert_angle,
                                                geometric_l.vert_angle))
        return base_poly_comb


def __search_in_street_names(text):
    is_street_pattern = re.match(STREET_PATERN, text, re.IGNORECASE)

    if is_street_pattern:
        return text in street_names or is_street_pattern.group(1) in street_names
    else:
        return text in street_names or (text + " street") in street_names


def compund_bboxes(first_bbox, last_bbox, panorama):
    """Combine 2 BoxInstances and update their data accordingly

    :param first_bbox: First box to combine
    :param last_bbox: Second box (Word will be to the right (next reading order) of the first)
    :param panorama: the panorama image from which the boxes were extracted
    :return: New and combined BoxInstance
    """

    new_bboxes = first_bbox.geometric.polygon.union(
        last_bbox.geometric.polygon).minimum_rotated_rectangle.exterior.coords[:-1]
    new_bboxes = np.round(np.array(sum(new_bboxes, ()), dtype=np.float32))

    new_char_scores = np.vstack((first_bbox.word.char_scores,
                                 last_bbox.word.char_scores))
    new_text = first_bbox.word.text + " " + last_bbox.word.text

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
    """Expand the WordInstances list to BoxInstances list which include geometric and color data

    :param twords: list of WordInstances extracted from the panorama
    :param panorama: the panorama image from which the words were extracted
    :return: list of BoxInstances
    """
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
