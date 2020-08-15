from charnet.modeling.postprocessing import WordInstance
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.affinity import scale
from fuzzywuzzy import process


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
        angle, line_len = __get_rect_properties(base_poly, horizontal)
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


def __get_rect_properties(base_poly, horizontal):
    # Exterior looks like this:
    # 3 ----- 2
    # |       |
    # 0 ----- 1
    point_a = base_poly.exterior.coords[0]
    point_b = base_poly.exterior.coords[1] if horizontal else base_poly.exterior.coords[3]
    angle = np.arctan2(point_b[1] - point_a[1],
                       point_b[0] - point_a[0])
    line_len = np.sqrt(((point_b[0] - point_a[0]) ** 2) + ((point_b[1] - point_a[1]) ** 2))
    return angle, line_len


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


def __compund_words(first_t, last_t, first_p, last_p, amprecent=False):
    new_bboxes = first_p.union(
        last_p).minimum_rotated_rectangle.exterior.coords[:-1]
    new_bboxes = np.round(np.array(sum(new_bboxes, ()), dtype=np.float32))

    new_char_scores = np.vstack((first_t.char_scores,
                                 last_t.char_scores))
    new_text = first_t.text + (" & " if amprecent else " ") + last_t.text

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


def analyze_extracted_words(twords):
    #less_twords = __remove_duplicates(twords, cut_off=90)
    # remove above 4 words?
    streets, others = __split_streets(twords)
    streets = __search_junctions(streets)
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


def __split_streets(twords):
    # find street names and define them as such
    # (hue)
    # "street"
    return twords, twords


def __search_junctions(streets, max_junctions=5):
    # find street junctions "regent st *&* barlet street"

    state = np.ones(len(streets), np.bool)
    word_polys = [__create_word_poly(b) for b in streets]
    combined_words = []

    current_junction = 0
    while state.any():
        base_idx = state.nonzero()[0][0]
        base_poly = __create_word_poly(streets[base_idx])
        base_poly_dialted = scale(base_poly, 3, 3)

        changed = False
        min_dist = 10000  # ridiculously large number
        closest_idx = None
        for idx in range(len(streets)):
            new_poly = word_polys[idx]
            if base_idx != idx and state[idx] and base_poly_dialted.intersects(new_poly) and \
                    current_junction < max_junctions:
                new_word = __compund_words(streets[base_idx], streets[idx],
                                           base_poly, new_poly, amprecent=True)
                streets[base_idx] = new_word
                state[idx] = False
                current_junction += 1
                changed = True
                break

        if not changed:
            state[base_idx] = False
            current_junction = 0
            combined_words.append(streets[base_idx])
    return combined_words


def __filter_others(others, cutoff_score=0.92):
    """apply filtering for words that weren't identified as streets.

        Currently uses naive cutoff scoring.

        :param others: a list of WordInstances to filter
        :param cutoff_score: the cutoff score below which the words will be deducted
        :return: a list of filtered WordInstances
    """
    return [b for b in others if b.text_score >= cutoff_score]



# plt.plot(*p.exterior.xy)