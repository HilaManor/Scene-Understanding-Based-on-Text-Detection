from charnet.modeling.postprocessing import WordInstance
import numpy as np
from shapely.geometry import Polygon

def concat_words(twords):
    """combines close words"""

    twords = concat_intersecting_words(twords)

    state = np.ones(len(twords), np.bool)
    word_polys = [__create_word_poly(b) for b in twords]
    new_words = []

    while state.any():
        base_idx = state.nonzero()[0][0]
        base_poly = __create_word_poly(twords[base_idx])
        changed = False
        min_dist = 50
        closest_idx = None
        for idx in range(len(twords)):
            new_poly = word_polys[idx]
            dist = new_poly.distance(base_poly)
            if base_idx != idx and state[idx] and dist < min_dist and \
                    abs(base_poly.bounds[0] - new_poly.bounds[0]) > 20:  # same row approx
                min_dist = dist
                closest_idx = idx

        if closest_idx:
            new_poly = word_polys[closest_idx]
            # if base first
            if base_poly.bounds[0] < new_poly.bounds[0]:  # or base_poly.bounds[1] < new_poly.bounds[1]:
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


def concat_intersecting_words(twords):
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