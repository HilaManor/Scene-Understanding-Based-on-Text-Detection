"""Warper module for charnet. Handles the first part of the 2nd algorithm stage: Extracting the
street-signs.

Functions for extracting words, using charnet, but on our windows of panorama framework"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
from shapely.geometry import Polygon
import torch
import cv2
from charnet.modeling.model import CharNet
from charnet.config import cfg
import numpy as np
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~


class CharNetRunner:
    """Warper for charnet. Handles extracting words on windows from the panorama

    Public Functions:
        get_absolute_window_words - extract words from a given window of the panorama
        clean_duplicate_words - remove duplicated from words extracted in the same window
        new_words_only - filter out words that have been extracted in previous windows

    Variables:
        config_file_path - path for the charnet config file
        charnet - charnet instance
    """
    def __init__(self, config_file):
        self.config_file_path = config_file

        # ----- CharNet configuration and initialization -----
        cfg.merge_from_file(config_file)
        cfg.freeze()
        print(cfg)

        self.charnet = CharNet()
        self.charnet.load_state_dict(torch.load(cfg.WEIGHT))
        self.charnet.eval()
        if torch.cuda.is_available():
            print("[*] Using cuda!")
            self.charnet.cuda()

    def get_absolute_window_words(self, pano_windows, window):
        """Get the words of the given window.

        Every word position is relative to the whole panorama, i.e. absolute position

        :param pano_windows: ImageWindows instance of the panorama
        :param window: Window instance of the current window, must be a member of the pano_windows
        :return: WordsInstance list of recognized words in the current window
        """
        words = []
        im, scale_w, scale_h, window_w, window_h = self.__resize(window.im)
        with torch.no_grad():
            # char_bboxes, char_scores, word_instances = ...
            _, _, word_instances = self.charnet(im, scale_w, scale_h, window_w, window_h)

        for word in word_instances:
            # To combat google's watermark of street-view messing with the words
            if word.text == 'GOOGLE':
                continue
            old_word_bbox = word.word_bbox.copy()
            # update absolute position
            word.word_bbox[::2] = [x_coord + window.pos_x for x_coord in word.word_bbox[::2]]
            word.word_bbox[1::2] = [y_coord + window.pos_y for y_coord in word.word_bbox[1::2]]
            word_abs = word
            # open a new window for near-border words
            if self.__word_is_near_border(old_word_bbox, 50, window_w, window_h):
                zoom_w = pano_windows.get_window_at_pos(word.word_bbox[0], word.word_bbox[1], 50)
                z_im, z_scale_w, z_scale_h, z_window_w, z_window_h = self.__resize(zoom_w.im)
                with torch.no_grad():
                    _, _, z_word_instances = self.charnet(z_im, z_scale_w, z_scale_h,
                                                          z_window_w, z_window_h)

                for z_word in z_word_instances:  # Swap only the word that intersects
                    z_word.word_bbox[::2] = [x_coord + zoom_w.pos_x for
                                             x_coord in z_word.word_bbox[::2]]
                    z_word.word_bbox[1::2] = [y_coord + zoom_w.pos_y for
                                              y_coord in z_word.word_bbox[1::2]]
                    if self._do_words_intersect(word, z_word):
                        word_abs = z_word  # save only the new word from the window
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
        """Checks if the bounding boxes of 2 words intersects (goes ontop of each-other)

        :param s_word: first WordInstance
        :param o_word: second WordInstance
        :return: Boolean of weather an intersection occurs
        """

        s_poly = Polygon([(x_pos, y_pos) for x_pos, y_pos in zip(s_word.word_bbox[::2],
                                                                 s_word.word_bbox[1::2])])
        o_poly = Polygon([(x_pos, y_pos) for x_pos, y_pos in zip(o_word.word_bbox[::2],
                                                                 o_word.word_bbox[1::2])])
        return s_poly.intersects(o_poly)

    @staticmethod
    def __word_is_near_border(bbox, margin, window_w, window_h):
        """Check if a given bounding box (of a word) is near the border of the window

        :param bbox: bounding box (of a word recognized by the network)
        :param margin: margin amount (in pixels) from the side of the window
        :param window_w: window's width
        :param window_h: window's height
        :return: Boolean of weather the word is near the border
        """
        #  [0][1]       [2][3]
        #
        #             [4][5]       [6][7]
        if bbox[2] > window_w - margin or bbox[6] > window_w - margin:
            return True
        if bbox[5] > window_h - margin or bbox[7] > window_h - margin:
            return True
        return False

    @staticmethod
    def clean_duplicate_words(words):
        """Remove words that appeared in multiple windows, but are in face the same instance

        This might happen when a word is near 2 margins, and so 2 windows will open for it
        seperatly, and so it will be recognized twice. The comparison is based on intersection,
        i.e. the absolute position of the word in the panorama

        :param words: WordsInstance list
        :return: filtered WordsInstance list
        """
        clean_words = []
        state = np.ones(len(words))
        while state.any():
            base_idx = state.nonzero()[0][0]
            for idx in range(len(words)):
                base_word = words[base_idx]
                new_word = words[idx]

                # if 2 different words (both still viable) intersect
                if idx != base_idx and state[idx] and \
                        CharNetRunner._do_words_intersect(base_word, new_word):
                    # decide if to keep left/right word
                    if base_word.text_score > new_word.text_score:
                        state[idx] = False
                    else:
                        state[base_idx] = False
                        break

            # if all other intersections were less accurate
            if state[base_idx]:
                clean_words.append(words[base_idx])
                state[base_idx] = False
        return clean_words

    @staticmethod
    def new_words_only(base_words, window_words):
        """Filter previously recognized words from this window's recognized words

        this are the same words position-wise.

        :param base_words: previously recognized words in a list of WordsInstance
        :param window_words: this window's list of WordsInstance recognized words
        :return: filtered WordsInstance list
        """
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
