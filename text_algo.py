import cv2
from pylab import *
from shapely.geometry import LineString
from shapely.affinity import scale
from fuzzywuzzy import process
import box_algo
from scipy import stats


def concat_words(tboxes):
    """combines close words"""
    print("[+] Connecting words...")
    # twords = __concat_intersecting_words(twords)
    tboxes = __concat_adjacent_words(tboxes, horizontal=True)
    tboxes = __concat_adjacent_words(tboxes, horizontal=False)
    return tboxes


def __concat_adjacent_words(tboxes, horizontal):
    state = np.ones(len(tboxes), np.bool)
    new_bboxes = []
    while state.any():
        base_idx = state.nonzero()[0][0]
        center = (tboxes[base_idx].geometric.polygon.centroid.x,
                  tboxes[base_idx].geometric.polygon.centroid.y)
        line_len = tboxes[base_idx].geometric.horiz_len if horizontal \
            else tboxes[base_idx].geometric.vert_len
        angle = tboxes[base_idx].geometric.horiz_angle if horizontal \
            else tboxes[base_idx].geometric.vert_angle
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
                new_bbox = box_algo.compund_bboxes(tboxes[base_idx], tboxes[closest_idx])
            else:  # if base last
                new_bbox = box_algo.compund_bboxes(tboxes[closest_idx], tboxes[base_idx])

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

    angle_check = abs(box_b.geometric.horiz_angle - box_a.geometric.horiz_angle) < angle_diff and \
                  abs(box_b.geometric.vert_angle - box_a.geometric.vert_angle) < angle_diff

    if cross_check:
        sf,pf = stats.ttest_ind(box_a.color_stats.color_data[0], box_b.color_stats.color_data[0],
                                equal_var=False)
        sk, pk = stats.kruskal(box_a.color_stats.color_data[0], box_b.color_stats.color_data[0])
        print("for words: %s, %s we got\n"
              "\tTTEST -> p=%.2f\n\tKRUKSAL -> p=%.2f" %
              (box_a.word.text, box_b.word.text, pf*100, pk*100))
        hue_check = abs(box_a.color_stats.hue_mean - box_b.color_stats.hue_mean) <= color_diff and \
                    box_a.color_stats.hue_std <= std_hue_limit and \
                    box_b.color_stats.hue_std <= std_hue_limit
        sat_check = abs(box_a.color_stats.sat_mean - box_b.color_stats.sat_mean) <= color_diff and \
                    box_a.color_stats.sat_std <= std_limit and \
                    box_b.color_stats.sat_std <= std_limit
        val_check = abs(box_a.color_stats.val_mean - box_b.color_stats.val_mean) <= color_diff and \
                    box_a.color_stats.val_std <= std_limit and \
                    box_b.color_stats.val_std <= std_limit
        color_check = hue_check or (sat_check and val_check)
        return cross_check and color_check# and angle_check
    else:
        return cross_check# and angle_check

# hue_mean = hue_mean
#         self.hue_std = hue_std
#         self.sat_mean = sat_mean
#         self.sat_std = sat_std
#         self.val_mean = val_mean
#         self.val_std = val_std
#         self.color_data = (hues, sats, vals)


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


# def __search_junctions(streets, max_junctions=5):
#     # find street junctions "regent st *&* barlet street"
#
#     state = np.ones(len(streets), np.bool)
#     word_polys = [__create_word_poly(b) for b in streets]
#     combined_words = []
#
#     current_junction = 0
#     while state.any():
#         base_idx = state.nonzero()[0][0]
#         base_poly = __create_word_poly(streets[base_idx])
#         base_poly_dialted = scale(base_poly, 3, 3)
#
#         changed = False
#         min_dist = 10000  # ridiculously large number
#         closest_idx = None
#         for idx in range(len(streets)):
#             new_poly = word_polys[idx]
#             if base_idx != idx and state[idx] and base_poly_dialted.intersects(new_poly) and \
#                     current_junction < max_junctions:
#                 new_word = __compund_words(streets[base_idx], streets[idx],
#                                            base_poly, new_poly, amprecent=True)
#                 streets[base_idx] = new_word
#                 state[idx] = False
#                 current_junction += 1
#                 changed = True
#                 break
#
#         if not changed:
#             state[base_idx] = False
#             current_junction = 0
#             combined_words.append(streets[base_idx])
#     return combined_words


# def __concat_intersecting_words(twords):
#     state = np.ones(len(twords), np.bool)
#     new_words = []
#     while state.any():
#         base_idx = state.nonzero()[0][0]
#         base_poly = __create_word_poly(twords[base_idx])
#         changed = False
#         for idx in range(len(twords)):
#             new_poly = word_polys[idx]
#             if base_idx != idx and state[idx] and base_poly.intersects(new_poly):
#                 # if base first
#                 if base_poly.bounds[0] < new_poly.bounds[0]:  # or base_poly.bounds[1] < new_poly.bounds[1]:
#                     new_word = __compund_words(twords[base_idx], twords[idx], base_poly, new_poly)
#                 else:  # if base last
#                     new_word = __compund_words(twords[idx], twords[base_idx], new_poly, base_poly)
#
#                 twords[base_idx] = new_word
#                 state[idx] = False
#                 changed = True
#                 break
#
#         if not changed:
#             state[base_idx] = False
#             new_words.append(twords[base_idx])
#     return new_words