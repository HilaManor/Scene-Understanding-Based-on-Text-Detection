"""This module handles the last part of the 2nd algorithm stage: Extracting the street-signs

extracted words are analyzed, combined into complete signs (sentences),
graded for street-sign-iness and the filtered"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
import cv2
from shapely.geometry import LineString
from fuzzywuzzy import process
import box_algo
import numpy as np
import re

# ~~~~~~~~~~~~~~~~~~~~~~~~ Constants ~~~~~~~~~~~~~~~~~~~~~~~
EXCLUSIONS = ['ONE WAY', 'STOP']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~


def concat_words(tboxes, panorama):
    """Combine close words that came from the same sign.

    :param tboxes: list of BoxInstances that were extracted from the panorama
    :param panorama: the panorama image from which the words originates (for color checks)
    :return: list of BoxInstances in which adjacent words were combined
    """
    """combines close words"""
    print("[+] Connecting words...")
    tboxes = __concat_adjacent_words(tboxes, panorama, horizontal=True)
    tboxes = __concat_adjacent_words(tboxes, panorama, horizontal=False)
    return tboxes


def analyze_extracted_words(tboxes, street_grade=85):
    """Grade, filter, and split the words into final street/non-street signs

    :param tboxes: list of BoxInstances that were extracted from the panorama
    :param street_grade: cutoff grade for a sign to be considered a street-sign
    :return: tuple of 2 lists of BoxInstances, one for street-signs and one for other signs
    """

    __update_grades(tboxes)
    less_tboxes = __remove_duplicates(tboxes, cut_off=85, street_grade=street_grade)
    streets, others = __split_streets(less_tboxes, street_grade=street_grade)
    others = __filter_others(others, cutoff_score=0.92)

    return streets, others


def __concat_adjacent_words(tboxes, panorama, horizontal):
    """Combine all adjacent words horizontally/vertically that are from the same sign

    :param tboxes: list of BoxInstances to connect words in
    :param panorama: the panorama image from which the words originates (for color checks)
    :param horizontal: Boolean for weather to connect horizontally or vertically
    :return: list of BoxInstances in which adjacent words were combined
    """

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
    """Checks if the 2 signs ("words") can come from the same sign ("sentence")

    Check is based on Geometric check (being on the same "line")
    and a color-background statistical check (hue/sat histogram is from the same distribution)

    :param box_a: sign uno (to match to)
    :param box_b: sign dos (to match)
    :param cont_right_line: a line that originates in the center of a and extend to the right,
                for a distance equal to the sing's length. the line is parallel to the word's angle
    :param cont_left_line: same, but to the left
    :param angle_diff: angle diff cutoff. Currently unused
    :return: Boolean - can be from the same sign?
    """

    # Check geomtry. the second sign must intersect on of the lines
    cross_check = box_b.geometric.polygon.crosses(cont_right_line) or \
                  box_b.geometric.polygon.crosses(cont_left_line)

    if cross_check:  # No point in checking others
        # unused
        angle_check = abs(box_b.geometric.horiz_angle - box_a.geometric.horiz_angle) < angle_diff \
                      and abs(box_b.geometric.vert_angle - box_a.geometric.vert_angle) < angle_diff

        # statistical tests
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

        # only 3 tests are used.
        color_check = (kl_div < 20000) + (correl > 0.5) + (bhattacharyya < 0.8)
        return color_check >= 2  # and angle_check
    return cross_check  # False


def __update_grades(tboxes):
    for box in tboxes:
        box.update_grade()


def __remove_duplicates(tboxes, cut_off=90, street_grade=90):
    """remove duplicated words that were found partially or fully, based on levinstein distance


        :param tboxes: a list of BoxInstances to filter
        :param cut_off: score below which the words will deemed "unsimilar" and won't be removed
        :type street_grade: street signs grade cutoff
        :return: a list of filtered WordInstances
    """

    # removes lonely streets-suffix such as "st"
    less_tboxes = [box for box in tboxes if
                   not bool(re.match('^' + box_algo.ST_SUFFIX + '$', box.word.text, re.I))]

    # chooses longer words first (e.g: "duane st" is better than "duane")
    s_wrds = sorted(less_tboxes, key=lambda x: (len(x.word.text), x.word.text_score), reverse=True)
    extracted_words = []
    while len(s_wrds):
        comp = s_wrds[0]
        choices = s_wrds[1:]
        similar_words = process.extractBests(comp, choices, processor=lambda x: x.word.text,
                                             limit=len(choices), score_cutoff=cut_off)
        extracted_words.append(comp)
        s_wrds.remove(comp)
        for sim in similar_words:
            # unless this is a street sign, delete it
            if sim[0].grade >= street_grade and \
                    not __same_street_sign_exists(sim[0], extracted_words,
                                                  street_grade=street_grade):
                extracted_words.append(sim[0])
            s_wrds.remove(sim[0])
    return [word for word in extracted_words if len(word.word.text.split()) <= 3]


def __same_street_sign_exists(box, base_boxes, street_grade=90):
    """Check if same street sign alreay exists, while not considering keywords - just names.

    The regex used checks that the name is exactly the same, and ignores for suffixes
    Remember that longer signs were extracted first, and so if one of the words have a suffix and
    the other not, it will not be the current checked word, but the one in the given list

    :param box: new street sign to check
    :param base_boxes: already extracted street signs
    :param street_grade: street grade to validate that it is in fact a street
    :return:
    """

    for b in base_boxes:
        if b.grade >= street_grade and \
                bool(re.match(box.word.text+'(\\s)?(?(1)'+box_algo.ST_SUFFIX+'|$)',
                              b.word.text, re.I)):
            return True
    return False


def __split_streets(tboxes, street_grade=75):
    """split the BoxInstances list to streets and non-streets based on grade cutoff

    :param tboxes: BoxInstances list of all signs
    :param street_grade: streets grade cutoff
    :return: tuple of 2 BoxInstances lists:  of street-signs and of non-street-signs
    """
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
    better_others = [b for b in others if b.word.text_score >= cutoff_score and b.word.text
                     not in EXCLUSIONS]
    longer_others = [b for b in better_others if len(b.word.text) > 3]
    return sorted(longer_others, key=lambda x: x.word.text_score, reverse=True)[:5]
