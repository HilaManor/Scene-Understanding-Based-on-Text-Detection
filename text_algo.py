import cv2
from shapely.geometry import LineString
from fuzzywuzzy import process
import box_algo
from scipy import stats
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


def _can_be_same_sign(box_a, box_b, cont_right_line, cont_left_line, angle_diff=0.1,
                      color_diff=10, std_hue_limit=30, std_limit=50):
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

        # hue_check = abs(
        #     box_a.color_stats.hue_mean - box_b.color_stats.hue_mean) <= color_diff and \
        #             box_a.color_stats.hue_std <= std_hue_limit and \
        #             box_b.color_stats.hue_std <= std_hue_limit
        # sat_check = abs(
        #     box_a.color_stats.sat_mean - box_b.color_stats.sat_mean) <= color_diff and \
        #             box_a.color_stats.sat_std <= std_limit and \
        #             box_b.color_stats.sat_std <= std_limit
        # val_check = abs(
        #     box_a.color_stats.val_mean - box_b.color_stats.val_mean) <= color_diff and \
        #             box_a.color_stats.val_std <= std_limit and \
        #             box_b.color_stats.val_std <= std_limit
        # color_check = hue_check or (sat_check and val_check)
        color_check = (kl_div < 20000) + (correl > 0.5) + (bhattacharyya < 0.8)
        return color_check >= 2 #and angle_check
    return cross_check  # False


def analyze_extracted_words(twords, panorama):
    # less_twords = __remove_duplicates(twords, cut_off=90)
    # TODO remove above 4 words?
    streets, others = __split_streets(twords, panorama)
    # streets = __search_junctions(streets)
    others = __filter_others(others, cutoff_score=0.92)
    return streets, others


def __remove_duplicates(twords, cut_off=90):
    """remove duplicated words that were found partially or fully, based on levinstein distance

        :param twords: a list of WordInstances to filter
        :param cut_off: score below which the words will deemed "unsimilar" and won't be removed
        :return: a list of filtered WordInstances
    """
    # remove_duplicates and half_duplicates
    s_wrds = sorted(twords, key=lambda x: len(x.text), reverse=True)
    extracted_words = []
    while len(s_wrds):
        comp = s_wrds[0]
        choices = s_wrds[1:]
        similar_words = process.extractBests(comp, choices, processor=lambda x: x.text,
                                             limit=len(choices), score_cutoff=cut_off)
        extracted_words.append(comp)
        s_wrds.remove(comp)
        for sim in similar_words:
            s_wrds.remove(sim[0])
    return extracted_words


def __split_streets(twords, panorama):
    length = len(twords)
    streets_signs = []
    other_signs = []
    # for word in twords:
    #     # check_word = re.findall(r'(.*\s(ST|WAY|STREET|AV|AVE|AVENUE|BD|BV|BVD|BOULEVARD|RD|ROAD))',
    #     #                         word.text, re.IGNORECASE)
    #     if check_word:
    #         # Found a street sign
    #         streets_signs.append(word)
    #     else:
    #         # Maybe not a street sign
    #
    #         other_signs.append(word)
    #Needs to check if there are more street signs via the histograms


    # find street names and define them as such
    # (hue)
    # "street"
    return twords, twords


def __filter_others(others, cutoff_score=0.92):
    """apply filtering for words that weren't identified as streets.

        Currently uses naive cutoff scoring.

        :param others: a list of WordInstances to filter
        :param cutoff_score: the cutoff score below which the words will be deducted
        :return: a list of filtered WordInstances
    """
    return [b for b in others if b.text_score >= cutoff_score]


# plt.plot(*p.exterior.xy)
