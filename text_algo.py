import cv2
from shapely.geometry import LineString
from fuzzywuzzy import process
import box_algo
import numpy as np

def concat_words(tboxes, panorama):
    """combines close words"""
    print("[+] Connecting words...")
    tboxes = __concat_adjacent_words(tboxes, panorama, horizontal=True)
    tboxes = __concat_adjacent_words(tboxes, panorama, horizontal=False)
    return tboxes


def __concat_adjacent_words(tboxes, panorama, horizontal):
    state = np.ones(len(tboxes), np.bool)
    new_bboxes = []
    while state.any():
        base_idx = state.nonzero()[0][0]
        center = (tboxes[base_idx].geometric.polygon.centroid.x,
                  tboxes[base_idx].geometric.polygon.centroid.y)
        line_len = tboxes[base_idx].geometric.horiz_len if horizontal else \
            tboxes[base_idx].geometric.vert_len
        angle = tboxes[base_idx].geometric.horiz_angle if horizontal else \
            tboxes[base_idx].geometric.vert_angle
        cont_right_line = LineString([(center[0], center[1]),
                                      (center[0] + line_len * np.cos(angle),
                                       center[1] + line_len * np.sin(angle))])
        cont_left_line = LineString([(center[0], center[1]),
                                     (center[0] + line_len * np.cos(angle + np.pi),
                                      center[1] + line_len * np.sin(angle + np.pi))])
        changed = False
        min_dist = 10000  # ridiculously large number
        closest_idx = None
        for idx in range(len(tboxes)):
            new_poly = tboxes[idx].geometric.polygon
            dist = new_poly.distance(tboxes[base_idx].geometric.polygon)
            if base_idx != idx and state[idx] and dist < min_dist and \
                _can_be_same_sign(tboxes[base_idx], tboxes[idx],
                                  cont_right_line, cont_left_line):
                min_dist = dist
                closest_idx = idx

        if closest_idx:
            new_poly = tboxes[closest_idx].geometric.polygon
            # if base first
            if new_poly.crosses(cont_right_line):
                new_bbox = box_algo.compund_bboxes(tboxes[base_idx], tboxes[closest_idx], panorama)
            else:  # if base last
                new_bbox = box_algo.compund_bboxes(tboxes[closest_idx], tboxes[base_idx], panorama)

            tboxes[base_idx] = new_bbox
            state[closest_idx] = False
            changed = True

        if not changed:
            state[base_idx] = False
            new_bboxes.append(tboxes[base_idx])
    return new_bboxes


def _can_be_same_sign(box_a, box_b, cont_right_line, cont_left_line, angle_diff=0.1):
    cross_check = box_b.geometric.polygon.crosses(cont_right_line) or \
                  box_b.geometric.polygon.crosses(cont_left_line)

    if cross_check:  # No point in checking others
        angle_check = abs(box_b.geometric.horiz_angle - box_a.geometric.horiz_angle) < angle_diff \
                      and abs(box_b.geometric.vert_angle - box_a.geometric.vert_angle) < angle_diff
        kl_div = cv2.compareHist(box_a.color_stats.hist, box_b.color_stats.hist,
                                 cv2.HISTCMP_KL_DIV)
        chi_sqr = cv2.compareHist(box_a.color_stats.hist, box_b.color_stats.hist,
                                  cv2.HISTCMP_CHISQR)
        chi_sqr_alt = cv2.compareHist(box_a.color_stats.hist, box_b.color_stats.hist,
                                      cv2.HISTCMP_CHISQR_ALT)
        bhattacharyya = cv2.compareHist(box_a.color_stats.hist, box_b.color_stats.hist,
                                        cv2.HISTCMP_BHATTACHARYYA)
        correl = cv2.compareHist(box_a.color_stats.hist, box_b.color_stats.hist,
                                 cv2.HISTCMP_CORREL)

        color_check = (kl_div < 20000) + (correl > 0.5) + (bhattacharyya < 0.8)
        return color_check >= 2 #and angle_check
    return cross_check  # False


def analyze_extracted_words(tboxes, panorama, street_grade=85):
    __update_grades(tboxes)
    less_tboxes = __remove_duplicates(tboxes, cut_off=85, street_grade=street_grade)
    streets, others = __split_streets(less_tboxes, street_grade=street_grade)
    others = __filter_others(others, cutoff_score=0.92)

    return streets, others

def __update_grades(tboxes):
    for box in tboxes:
        box.update_grade()

def __remove_duplicates(tboxes, cut_off=90, street_grade=90):
    """remove duplicated words that were found partially or fully, based on levinstein distance

        :param twords: a list of WordInstances to filter
        :param cut_off: score below which the words will deemed "unsimilar" and won't be removed
        :return: a list of filtered WordInstances
    """
    # remove_duplicates and half_duplicates
    s_wrds = sorted(tboxes, key=lambda x: len(x.word.text), reverse=True)
    extracted_words = []
    while len(s_wrds):
        comp = s_wrds[0]
        choices = s_wrds[1:]
        similar_words = process.extractBests(comp, choices, processor=lambda x: x.word.text,
                                             limit=len(choices), score_cutoff=cut_off)
        extracted_words.append(comp)
        s_wrds.remove(comp)
        for sim in similar_words:
            s_wrds.remove(sim[0])
    return [word for word in extracted_words if len(word.word.text.split()) <= 3]


def __split_streets(tboxes, street_grade=75):
    streets = []
    others = []
    for box in tboxes:
        if box.grade >= street_grade:
            streets.append(box)
        else:
            others.append(box)
    return streets, others


def __filter_others(others, cutoff_score=0.92):
    """apply filtering for words that weren't identified as streets.

        Currently uses naive cutoff scoring.

        :param others: a list of WordInstances to filter
        :param cutoff_score: the cutoff score below which the words will be deducted
        :return: a list of filtered WordInstances
    """
    return [b for b in others if b.word.text_score >= cutoff_score]

# plt.plot(*p.exterior.xy)
