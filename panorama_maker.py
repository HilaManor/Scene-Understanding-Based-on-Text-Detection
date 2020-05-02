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
cv2.ocl.setUseOpenCL(False)

# Constants

class PanoramaMaker(Object):
    def __init__(self):
        self.__counter = 0 # todo do I need that counter? maybe for later use
        self.__photos = [] # empty list named photos
        self.__featuresKeys = [] # empty list named features
        self.__check_first = 0


        pass

    def add_photo(self, image, featuresKeys=None): #images pre-processing
            # read image
            photos.append(image) #needs to check the syntax
            image_gray = cv2.cvtColor(photos[counter2], cv2.COLOR_RGB2GRAY) #now our photo in gray for data extraction
            key_points, photo_features = detectAndDescribe(image_gray, method=feature_extractor) # todo - method?
            featuresKeys.append(key_points, photo_features) # adding the photo information to the list as a tuple
            if check_first < count(featuresKeys()[1]):
                check_first = counter
            counter++  # todo - again, do I need that counter?
        
        pass

        # once you have all images processing
        def create_panorama(self):
            # panorama =

            # search

            return panorama




    def __detectAndDescribe(self, image, method=None):
        # Compute key points and feature descriptors using an specific method

        assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

        # detect and extract features from the image
        if method == 'sift':
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif method == 'surf':
            descriptor = cv2.xfeatures2d.SURF_create()
        elif method == 'brisk':
            descriptor = cv2.BRISK_create()
        elif method == 'orb':
            descriptor = cv2.ORB_create()

        # get keypoints and descriptors
        (kps, features) = descriptor.detectAndCompute(image, None)

        return (kps, features)


    def __createMatcher(self, method, crossCheck): #create a matcher - NORM is chosen per extractor
        "Create and return a Matcher Object"

        if method == 'sift' or method == 'surf':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        elif method == 'orb' or method == 'brisk':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        return bf


    def __matchKeyPointsBF(self, featuresKeys, method):
        bf = createMatcher(method, crossCheck=True)

        # Match descriptors.
        best_matches = bf.match(featuresA, featuresB)

        # Sort the features in order of distance.
        # The points with small distance (more similarity) are ordered first in the vector
        rawMatches = sorted(best_matches, key=lambda x: x.distance)
        print("Raw matches (Brute force):", len(rawMatches))
        return rawMatches















    #     """ blah balh ablhabla alhablah
    #
    #     :param self:
    #     :param a: age
    #     :param b: birthday
    #
    #     :return: blah balh
    #     """
    #     pass
    #
    # def create_panorama(self):
    #
    #     return panorama_img

