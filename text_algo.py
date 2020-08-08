from charnet.modeling.postprocessing import WordInstance
import numpy as np
from shapely.geometry import Polygon

def concat_words(twords):
    """combines close words"""
    return twords
    # state = np.zeros(len(twords), np.bool)
    # word_polys = [Polygon([(b.word_bbox[0], b.word_bbox[1]), (b.word_bbox[2], b.word_bbox[3]),
    #                        (b.word_bbox[4], b.word_bbox[5]), (b.word_bbox[6], b.word_bbox[7])])
    #               for b in twords]
    #
    # for base_idx in state.nonzero()[0]:
    #     base_poly = word_polys[idx]
    #     for idx in word_polys:
    #         if base_idx != idx and base_poly.intersects(word_polys[idx]):
    #             new_bboxes = base_poly.union(word_polys[idx]).minimum_rotated_rectangle.coords[:-1]
    #             new_bboxes =
    #             new_word = WordInstance(
    #                 word_bboxes[idx],
    #                 word_bbox_scores[idx],
    #                 text, text_score,
    #                 tmp_char_scores
    #             )