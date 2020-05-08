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
    def __init__(self, descriptor_type=DescriptorType.ORB, matcher_type=BF):
        self.__counter = 0 # todo do I need that counter? maybe for later use
        self.__photos = [] # empty list named photos
        self.__featuresKeys = [] # empty list named features
        self.__check_first = 0
        self.__descriptor_type = descriptor_type # todo - check: do I need here __ (as all - matcher)
        self.__descriptor_func = self.__choose_descriptor_type() # used in detectAndDescribe
        self.__matcher_type = matcher_type

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
        key_points, photo_features = self.__detectAndDescribe(image_gray) # getting information
        self.__featuresKeys.append(key_points, photo_features) # adding the photo's information to the list as a tuple
        if self.__check_first < len(photo_features): # for later use - who's will be first
            self.__check_first = self.__counter
        self.__counter += 1  # todo - again, do I need that counter?
        
        # once you have all images processing
        def create_panorama(self):

            i = 0
            checkArray = [1] * self.__counter # which photos are already included - will change to zero inside the loop
            while ( i < self.__counter ):
            # panorama =

            # it's gonna save each time the new photo until it's gonna be panoramic


            i += 1
            # search




            # return panorama
            pass

    def __detectAndDescribe(self, image):
        # Compute key points and feature descriptors using an specific method
        descriptor = self.__descriptor_func()
        # get keypoints and descriptors
        return descriptor.detectAndCompute(image, None)

    def __matcherType(self): # the importance is the default section (BF)
        if self.__matcher_type == BF
            return BF
        elif self.__matcher_type == KNN
                return KNN
        else:
                print("no such matcher, choosing BF")
                return BF

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

    def __matchKeyPoints(self): # todo - don't forget make it robust

        matcher = self.__matcherType()
        if matcher == BF:
            bf = createMatcher(method, crossCheck=True)
        elif matcher == KNN:
            bf = createMatcher(method, crossCheck=False)
        else:
            raise ("should not get here")  # We dealt with that previously, BF is defaulty

        i = 0
        best_matches = []
        while i < self.__counter: # todo I'll need here that array to know who's to continue look for
            if macther == bf:
            best_matches[i] = bf.match(self.__featuresKeys[i][1], self.__featuresKeys[???][1] )




        pass



    def matchKeyPointsBF(featuresA, featuresB, method): # it's theirs!!!
        bf = createMatcher(method, crossCheck=True)

        # Match descriptors.
        best_matches = bf.match(featuresA, featuresB)

        # Sort the features in order of distance.
        # The points with small distance (more similarity) are ordered first in the vector
        rawMatches = sorted(best_matches, key=lambda x: x.distance)
        print("Raw matches (Brute force):", len(rawMatches))
        return rawMatches

    def matchKeyPointsKNN(featuresA, featuresB, ratio, method): # it's theirs!
        bf = createMatcher(method, crossCheck=False)
        # compute the raw matches and initialize the list of actual matches
        rawMatches = bf.knnMatch(featuresA, featuresB, 2)
        print("Raw matches (knn):", len(rawMatches))
        matches = []

        # loop over the raw matches
        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * ratio:
                matches.append(m)
        return matches





















    def matchKeyPointsBF(self): # that's the old one, delete later
        bf = createMatcher(crossCheck=True)
        # Match descriptors.
b
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













