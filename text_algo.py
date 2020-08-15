from charnet.modeling.postprocessing import WordInstance
import numpy as np
import re
import cv2
from pylab import *
from shapely.geometry import Polygon, LineString



def concat_words(twords):
    """combines close words"""

    twords = __concat_intersecting_words(twords)

    twords = __concat_adjacent_words(twords, horizontal=True)
    twords = __concat_adjacent_words(twords, horizontal=False)
    return twords


def __concat_adjacent_words(twords, horizontal):
    state = np.ones(len(twords), np.bool)
    word_polys = [__create_word_poly(b) for b in twords]
    new_words = []
    while state.any():
        base_idx = state.nonzero()[0][0]
        base_poly = __create_word_poly(twords[base_idx])

        # Exterior looks like this:
        # 3 ----- 2
        # |       |
        # 0 ----- 1
        point_a = base_poly.exterior.coords[0]
        point_b = base_poly.exterior.coords[1] if horizontal else base_poly.exterior.coords[3]

        angle = np.arctan2(point_b[1] - point_a[1],
                           point_b[0] - point_a[0])

        line_len = np.sqrt(((point_b[0] - point_a[0]) ** 2) + ((point_b[1] - point_a[1]) ** 2))
        cont_right_line = LineString([(base_poly.centroid.x, base_poly.centroid.y),
                                      (base_poly.centroid.x + line_len * np.cos(angle),
                                       base_poly.centroid.y + line_len * np.sin(angle))])
        cont_left_line = LineString([(base_poly.centroid.x, base_poly.centroid.y),
                                     (base_poly.centroid.x + line_len * np.cos(angle + np.pi),
                                      base_poly.centroid.y + line_len * np.sin(angle + np.pi))])
        changed = False
        min_dist = 10000  # ridiculously large number
        closest_idx = None
        for idx in range(len(twords)):
            new_poly = word_polys[idx]
            dist = new_poly.distance(base_poly)
            if base_idx != idx and state[idx] and dist < min_dist and \
                    (new_poly.crosses(cont_right_line) or new_poly.crosses(cont_left_line)):
                min_dist = dist
                closest_idx = idx

        if closest_idx:
            new_poly = word_polys[closest_idx]
            # if base first
            if new_poly.crosses(cont_right_line):
                new_word = __compund_words(twords[base_idx], twords[closest_idx],
                                           base_poly, new_poly)
            else:  # if base last
                new_word = __compund_words(twords[closest_idx], twords[base_idx],
                                           new_poly, base_poly)

            twords[base_idx] = new_word
            state[closest_idx] = False
            changed = True

        if not changed:
            state[base_idx] = False
            new_words.append(twords[base_idx])
    return new_words


def __concat_intersecting_words(twords):
    state = np.ones(len(twords), np.bool)
    word_polys = [__create_word_poly(b) for b in twords]
    new_words = []
    while state.any():
        base_idx = state.nonzero()[0][0]
        base_poly = __create_word_poly(twords[base_idx])
        changed = False
        for idx in range(len(twords)):
            new_poly = word_polys[idx]
            if base_idx != idx and state[idx] and base_poly.intersects(new_poly):
                # if base first
                if base_poly.bounds[0] < new_poly.bounds[0]:  # or base_poly.bounds[1] < new_poly.bounds[1]:
                    new_word = __compund_words(twords[base_idx], twords[idx], base_poly, new_poly)
                else:  # if base last
                    new_word = __compund_words(twords[idx], twords[base_idx], new_poly, base_poly)

                twords[base_idx] = new_word
                state[idx] = False
                changed = True
                break

        if not changed:
            state[base_idx] = False
            new_words.append(twords[base_idx])


    return new_words


def __compund_words(first_t, last_t, first_p, last_p):
    new_bboxes = first_p.union(
        last_p).minimum_rotated_rectangle.exterior.coords[:-1]
    new_bboxes = np.round(np.array(sum(new_bboxes, ()), dtype=np.float32))

    new_char_scores = np.vstack((first_t.char_scores,
                                 last_t.char_scores))
    new_text = first_t.text + " " + last_t.text

    new_word = WordInstance(
        new_bboxes,
        np.average([first_t.word_bbox_score, last_t.word_bbox_score]),
        new_text,
        np.average([first_t.text_score, last_t.text_score]),
        new_char_scores
    )
    return new_word

def __create_word_poly(b):
    return Polygon([(b.word_bbox[0], b.word_bbox[1]), (b.word_bbox[2], b.word_bbox[3]),
             (b.word_bbox[4], b.word_bbox[5]), (b.word_bbox[6], b.word_bbox[7])])


def analyze_extracted_words(twords, panorama):
    twords = __remove_duplicates(twords)
    streets, others = __split_streets(twords, panorama)
    streets = __search_junctions(streets)
    others = __filter_others(others)
    return streets, others


def __remove_duplicates(twords):
    # remove_duplicates and half_duplicates
    s_wrds = sorted(twords, key=lambda x: len(x.text), reverse=True)

    return twords


def __split_streets(twords, panorama):
    length = len(twords)
    streets_signs = []
    other_signs = []
    for word in twords:
        check_word = re.findall(r'(.*\s(ST|WAY|STREET|AV|AVE|AVENUE|BD|BV|BVD|BOULEVARD|RD|ROAD))',
                                word.text, re.IGNORECASE)
        if check_word:
            # Found a street sign
            streets_signs.append(word)
        else:
            # Maybe not a street sign
            mask = np.zeros((panorama.shape[0], panorama.shape[1]), dtype=np.bool)
            poly = np.array(word.word_bbox).reshape((1, 4, 2))
            cv2.fillPoly(mask, poly, True)
            bins = 16
            range = (0, 255)
            hsv_img = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
            hue_img = hsv_img[:, :, 0]
            saturation_img = hsv_img[:, :, 1]
            value_img = hsv_img[:, :, 2]
            hist(hue_img[mask], bins=bins, range=range)
            hist(saturation_img[mask], bins=bins, range=range)
            hist(value_img[mask], bins=bins, range=range)


            other_signs.append(word)

    #Needs to check if there are more street signs via the histograms





    # find street names and define them as such
    # (hue)
    # "street"
    return twords


def __search_junctions(streets):
    # find street junctions "regent st *&* barlet street"
    return streets


def __filter_others(others):
    # from not-street, keep only those with high certainty
    return others
