# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# todo - for now the algorithm using BruteForce rather than KNN 
#
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
from enum import Enum
cv2.ocl.setUseOpenCL(False)

# Constants

class DescriptorType(Enum):
    ORB = 1
    BRISK = 2
    SURF = 3
    SIFT = 4
    AKAZE = 5


class PanoramaMaker:
    def __init__(self, descriptor_type=DescriptorType.ORB):
        self.__counter = 0 # todo do I need that counter? maybe for later use
        self.__photos = [] # empty list named photos
        self.__featuresKeys = [] # empty list named features
        self.__check_first = 0
        self.__descriptor_type = descriptor_type
        self.__descriptor_func = self.__choose_descriptor_type()

    def __choose_descriptor_type(self):
        if self.__descriptor_type == DescriptorType.ORB:
            return cv2.ORB.create
        elif self.__descriptor_type == DescriptorType.AKAZE:
            try:
                return cv2.AKAZE_create
            except AttributeError:
                print("AKAZE not available, choosing ORB")
                return cv2.ORB.create
        elif self.__descriptor_type == DescriptorType.BRISK:
            try:
                return cv2.BRISK_create
            except AttributeError:
                print("BRISK not available, choosing ORB")
                return cv2.ORB.create
        elif self.__descriptor_type == DescriptorType.SIFT:
            try:
                return cv2.xfeatures2d_SIFT.create
            except AttributeError:
                print("SIFT not available, choosing ORB")
                return cv2.ORB.create
        elif self.__descriptor_type == DescriptorType.SURF:
            try:
                return cv2.xfeatures2d_SURF.create
            except AttributeError:
                print("SURF not available, choosing ORB")
                return cv2.ORB.create
        else:
            print("no such descriptor, choosing ORB")
            return cv2.ORB.create

    def add_photo(self, image): #images pre-processing
        # read image
        self.__photos.append(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #now our photo in gray for data extraction
        key_points, photo_features = self.__detectAndDescribe(image_gray)
        self.__featuresKeys.append(key_points, photo_features) # adding the photo information to the list as a tuple
        if self.__check_first < len(photo_features):
            self.__check_first = self.__counter
        self.__counter += 1  # todo - again, do I need that counter?
        
        # once you have all images processing
        def create_panorama(self):
            # panorama =

            # search

            # return panorama
            pass



    def __detectAndDescribe(self, image):
        # Compute key points and feature descriptors using an specific method
        descriptor = self.__descriptor_func()
        # get keypoints and descriptors
        return descriptor.detectAndCompute(image, None)

    def __createMatcher(self, crossCheck):
        # creates and returns a matcher object
        # after feature-detection-description, feature matching is performed by using L2 norm
        # for string-based descriptors, and Hamming distance for binary descriptors
        if self.__descriptor_type == DescriptorType.SIFT or\
                self.__descriptor_type == DescriptorType.SURF:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        elif self.__descriptor_type == DescriptorType.ORB or self.__descriptor_type ==\
                DescriptorType.BRISK or self.__descriptor_type == DescriptorType.AKAZE:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        else:
            raise("should not get here")  # We dealt with that previously, ORB is defaulty
        return bf






    def __matchKeyPoints(self, featuresKeys, check_first, counter): # todo needs to add matcher object

        if matcher_object == BF # todo needs to add BF and KNN
            rawmatch = self.__matchKeyPointsBF(featuresKeys, )



    def matchKeyPointsBF(self, featuresKeys, check_first, counter):
        bf = createMatcher(crossCheck=True)
        # Match descriptors.

        index_a = check_first
        index = 0
        best_matches[[]] # todo - is that ok?
        raw_match[]
        for index < counter
            if index_a != index
                best_matches[index_a][index] = bf.match(featuresKeys[index_a], featuresKeys[index])
                # sort the features in order of distance
                # The points with small distance (more similarity) are ordered first in the vec
                raw_Match.append = sorted(best_matches[index_a][index], key=lambda x: x.distance)





            return rawMatches


s
        #
        # # loop over the raw matches
        # for m, n in rawMatches:
        #     # ensure the distance is within a certain ratio of each
        #     # other (i.e. Lowe's ratio test)
        #     if m.distance < n.distance * ratio:
        #         matches.append(m)
        # return matches













