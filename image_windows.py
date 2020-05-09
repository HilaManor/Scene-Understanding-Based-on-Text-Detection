"""Utility functions for getting images to run on from a panorama image.

class ImageWindows - wrapper for a big window functions
class Window - wrapper for functions on a signle window
"""

# Imports
import numpy as np


class ImageWindows:
    """Utility class for operating on sub-images of a given panorama.

    functions:
    __iter__ - returns all windows in the inited panorama
    get_window_at_pos - get a special window from the panorama

    variables:
    panorama - the inited panorama
    """

    def __init__(self, panorama, size_tpl=None, input_size_cfg=2280):
        """Create Object for a panorama.

        :param panorama: input panorama image (cv2)
        :param size_tpl: pre-determined size for window. otherwise will calculate optimal size
        :param input_size_cfg: input size that charnet is configured for, for calculating size
        """
        self.panorama = panorama
        self.__panorama_h = self.panorama.shape[0]
        self.__panorama_w = self.panorama.shape[1]

        self.__input_size_cfg = input_size_cfg
        self.__windows_size = size_tpl
        if not size_tpl:
            self.__windows_size = self.__calculate_window_size()

    def __iter__(self):
        """Create an iterator for all the windows in the panorama.

        Will not concatenate smaller window with previous ones (in edges):
            e.g.:  [[1,2,3],        [[1,2],     [[3],
                    [4,5,6],  -->    [4,5],   ,  [6],
                    [7,8,9]]         [7,8]]      [9]]

        :return: One window of an image
        """
        for y in range(0, self.__panorama_h, self.__windows_size[1]):
            for x in range(0, self.__panorama_w, self.__windows_size[0]):
                y_end = y + self.__windows_size[1]
                x_end = x + self.__windows_size[0]
                yield Window(x, y, x_end, y_end, self.panorama[y:y_end, x:x_end])

    def __len__(self):
        """Return the amount of all the windows in the panorama."""
        return len(range(0, self.__panorama_h, self.__windows_size[1])) * len(range(0, self.__panorama_w, self.__windows_size[0]))

    def get_window_at_pos(self, around_pos_x, around_pos_y, margin=0):
        """Return from the panorama a window starting from the given x,y, considering margins.

        :param around_pos_x: starting window x (width) coordinate
        :param around_pos_y: starting window y (height) coordinate
        :param margin: margin from the position to go up-left (might be useful for giving this func
                        directly the words coordinate, and pad so we won't miss it
        :return: normal window sized image from the panorama
        """
        x_start = int(around_pos_x) - margin
        y_start = int(around_pos_y) - margin
        x_start = 0 if x_start < 0 else x_start
        y_start = 0 if y_start < 0 else y_start
        y_end = y_start + self.__windows_size[1]
        x_end = x_start + self.__windows_size[0]
        return Window(x_start, y_start, x_end, y_end, self.panorama[y_start:y_end, x_start:x_end])

    def __calculate_window_size(self, upsacle_boundary=33):
        """Calculate the window's size in pixels.

        Given (via cfg) the ideal window size, calculates actual size by deciding how to split the
        rows an columns, based on given boundary.

        e.g.: for window size 40 and upsacle_boundary=40, 81x79 image will give (41,40)
    
        :param upsacle_boundary: precent of ideal window size over which to open a new chunk for
                                remaining data (under ideal window size.)
        :return: Tuple(Calculated window x size, Calculated window y size)
        """

        # If upscale_boundary is big, then if half of the cfg size is left - open new chunk
        if upsacle_boundary >= 50:
            cols_chunks_num = int(np.round(self.__panorama_w / self.__input_size_cfg))
            rows_chunks_num = int(np.round(self.__panorama_h / self.__input_size_cfg))
        else:  # Open new chunk if the remaining margin is bigger than given percentage
            dist_boundary = self.__input_size_cfg * upsacle_boundary // 100
            cols_chunks_num = self.__panorama_w // self.__input_size_cfg
            cols_chunks_num += (self.__panorama_w % self.__input_size_cfg > dist_boundary)

            # rows will be half length
            rows_chunks_num = self.__panorama_h // (self.__input_size_cfg // 2)
            rows_chunks_num += (self.__panorama_h % (self.__input_size_cfg / 2) > dist_boundary)

        actual_wind_col_size = (self.__panorama_w // cols_chunks_num) + 1
        actual_wind_row_size = (self.__panorama_h // rows_chunks_num) + 1

        return actual_wind_col_size, actual_wind_row_size


class Window:
    """Utility class for holding a window's attributes."""
    def __init__(self, pos_x, pos_y, end_x, end_y, im):
        """Create a window.

        :param pos_x: the window absolute x starting coords in the panorama from which it was taken
        :param pos_y: the window absolute y starting coords in the panorama from which it was taken
        :param end_x: the window absolute x ending coords in the panorama from which it was taken
        :param end_y: the window absolute y ending coords in the panorama from which it was taken
        :param im: the window's image (cv2)
        """
        self.pos = (pos_x, pos_y)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.im = im
        self.len_x = end_x - pos_x
        self.len_y = end_y - pos_y

