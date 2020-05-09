from shapely.geometry import Polygon

import torch
import cv2
from charnet.modeling.model import CharNet
from charnet.config import cfg

class CharNetRunner:
    def __init__(self, config_file):
        self.config_file_path = config_file

        # ----- CharNet configuration and initialization -----
        cfg.merge_from_file(config_file)
        cfg.freeze()
        print(cfg)

        self.charnet = CharNet()
        self.charnet.load_state_dict(torch.load(cfg.WEIGHT))
        self.charnet.eval()
        self.charnet.cuda()

    def get_absolute_window_words(self, pano_windows, window):
        words = []
        im, scale_w, scale_h, window_w, window_h = self.__resize(window.im)
        with torch.no_grad():  # TODO CHECK
            # char_bboxes, char_scores, word_instances
            _, _, word_instances = self.charnet(im, scale_w, scale_h, window_w, window_h)

        for word in word_instances:
            if word.text == 'GOOGLE':
                continue
            old_word_bbox = word.word_bbox.copy()
            # update absolute position
            word.word_bbox[::2] = [x_coord + window.pos_x for x_coord in word.word_bbox[::2]]
            word.word_bbox[1::2] = [y_coord + window.pos_y for y_coord in word.word_bbox[1::2]]
            if not self.__word_is_near_border(old_word_bbox, 50, window_w, window_h):
                word_abs = word
            else:
                zoom_w = pano_windows.get_window_at_pos(word.word_bbox[0], word.word_bbox[1], 50)
                z_im, z_scale_w, z_scale_h, z_window_w, z_window_h = self.__resize(zoom_w.im)
                with torch.no_grad():
                    _, _, z_word_instances = self.charnet(z_im, z_scale_w, z_scale_h,
                                                          z_window_w, z_window_h)

                for z_word in z_word_instances:  # Swap only the word that intersects
                    z_word.word_bbox[::2] = [x_coord + zoom_w.pos_x for x_coord in z_word.word_bbox[::2]]
                    z_word.word_bbox[1::2] = [y_coord + zoom_w.pos_y for y_coord in z_word.word_bbox[1::2]]
                    if self._do_words_intersect(word, z_word):
                        word_abs = z_word
                        break

            words.append(word_abs)
        return words

    @staticmethod
    def __resize(im, size=cfg.INPUT_SIZE):
        h, w, _ = im.shape
        scale = max(h, w) / float(size)
        image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
        image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
        scale_h = float(h) / image_resize_height
        scale_w = float(w) / image_resize_width
        im = cv2.resize(im, (image_resize_width, image_resize_height),
                              interpolation=cv2.INTER_LINEAR)
        return im, scale_w, scale_h, w, h

    @staticmethod
    def _do_words_intersect(s_word, o_word):
        s_poly = Polygon([(x_pos, y_pos) for x_pos, y_pos in zip(s_word.word_bbox[::2],
                                                                 s_word.word_bbox[1::2])])
        o_poly = Polygon([(x_pos, y_pos) for x_pos, y_pos in zip(o_word.word_bbox[::2],
                                                                 o_word.word_bbox[1::2])])
        return s_poly.intersects(o_poly)

    @staticmethod
    def __word_is_near_border(bbox, margin, window_w, window_h):
        #  [0][1]       [2][3]
        #
        #             [4][5]       [6][7]
        if bbox[2] > window_w - margin or bbox[6] > window_w - margin:
            return True
        if bbox[5] > window_h - margin or bbox[7] > window_h - margin:
            return True
        return False

    @staticmethod
    def new_words_only(base_words, window_words):
        new_words = []
        if base_words:
            for wword in window_words:
                dont_include = False
                for bword in base_words:
                    if CharNetRunner._do_words_intersect(wword, bword):
                        dont_include = True
                        break
                if not dont_include:
                    new_words.append(wword)
        else:
            new_words = window_words

        return new_words
