# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Imports
import numpy as np


class ImageWindows:
    def __init__(self, panorama, size_tpl=None, input_size_cfg=2280):
        self.panorama = panorama
        self.panorama_h = self.panorama.shape[0]
        self.panorama_w = self.panorama.shape[1]

        self.__input_size_cfg = input_size_cfg
        self.__windows_size = size_tpl
        if not size_tpl:
            self.__windows_size = self.__calculate_window_size()

    def __iter__(self):
        """ Iterator for all the windows in the panorama.
        Will not concatenate smaller window with previous ones (in edges):
            e.g.:  [[1,2,3],        [[1,2],     [[3],
                    [4,5,6],  -->    [4,5],   ,  [6],
                    [7,8,9]]         [7,8]]      [9]]

        :return: One window of an image
        """
        for y in range(0, self.panorama_h, self.__windows_size[1]):
            for x in range(0, self.panorama_w, self.__windows_size[0]):
                y_end = y + self.__windows_size[1]
                x_end = x + self.__windows_size[0]
                yield self.panorama[y:y_end, x:x_end]

    def get_window_at_pos(self, around_pos_tpl):  # TODO around/left side?

        pass

    def __calculate_window_size(self, upsacle_boundary=33):
        """ Calculates the window's size in pixels.
        Given (via cfg) the ideal window size, calculates actual size by deciding how to split the
        rows an columns, based on given boundary.

        e.g.: for window size 40 and upsacle_boundary=40, 81x79 image will give (41,40)
    
        :param upsacle_boundary: precent of ideal window size over which to open a new chunk for
                                remaining data (under ideal window size.)
        :return: Tuple(Calculated window x size, Calculated window y size)
        """

        # If upscale_boundary is big, then if half of the cfg size is left - open new chunk
        if upsacle_boundary >= 50:
            cols_chunks_num = int(np.round(self.panorama_w / self.__input_size_cfg))
            rows_chunks_num = int(np.round(self.panorama_h / self.__input_size_cfg))
        else:  # Open new chunk if the remaining margin is bigger than given percentage
            dist_boundary = self.__input_size_cfg * upsacle_boundary // 100
            cols_chunks_num = self.panorama_w // self.__input_size_cfg
            cols_chunks_num += (self.panorama_w % self.__input_size_cfg > dist_boundary)

            # rows will be half length
            rows_chunks_num = self.panorama_h // (self.__input_size_cfg // 2)
            rows_chunks_num += (self.panorama_h % (self.__input_size_cfg / 2) > dist_boundary)

        actual_wind_col_size = (self.panorama_w // cols_chunks_num) + 1
        actual_wind_row_size = (self.panorama_h // rows_chunks_num) + 1

        return actual_wind_col_size, actual_wind_row_size
