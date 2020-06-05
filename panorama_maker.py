# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This scripts is creating panoramic photo from multiply sub-photos
# Robust - has several options for descriptorType and two matchers options
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Imports
import cv2
import numpy as np
import imutils
from enum import Enum
cv2.ocl.setUseOpenCL(False)


class DescriptorType(Enum):
    ORB = 1
    BRISK = 2
    SURF = 3
    SIFT = 4
    AKAZE = 5

class MatcherType(Enum):
    BF = 1
    KNN = 2

class PanoramaMaker:
    def __init__(self, descriptor_type=DescriptorType.ORB, matcher_type=MatcherType.BF, ratio=0.75, reprojThresh=4):
        self.__counter = 0 # todo do I need that counter? maybe for later use
        self.__photos = [] # empty list named photos
        self.__featuresKeys = [] # empty list named features
        self.__counter = 0  # todo do I need that counter? maybe for later use
        self.__photos = []
        self.__featuresKeys = []
        self.__check_first = 0
        self.__descriptor_type = descriptor_type # todo - check: do I need here __ (as all - matcher)
        self.__descriptor_func = self.__choose_descriptor_type() # used in detectAndDescribe
        self.__matcher_type = matcher_type
        self.__ratio = ratio
        self.__reprojThresh = reprojThresh
        self.__homographies = []

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
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #for data extraction
        key_points, photo_features = self.__detectAndDescribe(image_gray)  # getting information
        self.__featuresKeys.append((key_points, photo_features)) # adding the photo's information to the list as a tuple
        if self.__check_first < len(photo_features):  # for later use - who's will be first
            self.__check_first = self.__counter
        self.__counter += 1
        
    # once you have all images processing
    def create_panorama(self):
        self.__homographies = self.__reorder()

        # strating the panorama with the photo which helds mots features
        panorama = self.__photos[self.__check_first]
        panorama_kps, panorama_features = self.__featuresKeys[self.__check_first]
        del self.__photos[self.__check_first]
        del self.__featuresKeys[self.__check_first]
        self.__counter -= 1

        # loop for all other photos
        while self.__counter:
            # Who's best match: by the matches length
            best_index, matches = self.__get_best_match(panorama_features)

            # For the best, we'll find homograpghy
            homography_result = self.__get_homography(matches, panorama_kps, panorama_features, best_index, self.__reprojThresh) # B in panorama
            if not homography_result:
                raise ( "No homograpghy found!" ) # todo: then what? continue to the other photos?

            # warp and stitch
            matches, H, status = homography_result
            # The stitched photo is the next one
            panorama = self.__warp_and_stitch(H, best_index, panorama)
            # delete used photo
            del self.__photos[best_index]
            del self.__featuresKeys[best_index]
            self.__counter -= 1
            panorama_gray = cv2.cvtColor(panorama.astype(np.uint8), cv2.COLOR_RGB2GRAY)  # for data extraction
            panorama_kps, panorama_features = self.__detectAndDescribe(panorama_gray)  # getting information on new panorama
        return panorama.astype(np.uint8)

    def __reorder(self):
        print('[+] Getting photo order...')
        ordered_photos = [self.__photos[self.__check_first]]
        ordered_featureKeys = [self.__featuresKeys[self.__check_first]]
        homographies = []
        del self.__photos[self.__check_first]
        del self.__featuresKeys[self.__check_first]

        while len(self.__photos):
            best_i_r = -1
            best_i_l = -1
            H_r = None
            H_l = None
            best_match_r = None
            best_match_l = None
            # whos the best from the left
            ignore_l_i = []
            while best_i_l == -1 and len(ignore_l_i) < len(self.__photos):
                best_i_l, best_match_l = self.__get_best_match_from_photos(ordered_featureKeys[0][1], ignore_l_i)
                H_l, status = self.__get_homography(best_match_l, ordered_featureKeys[0][0],
                                                    best_i_l, self.__reprojThresh)
                # not really from the left match
                if H_l[0, 2] >= 0:
                    ignore_l_i.append(best_i_l)
                    best_i_l = -1  # continue to search

            # whos the best from the right
            ignore_r_i = []
            while best_i_r == -1 and len(ignore_r_i) < len(self.__photos):
                best_i_r, best_match_r = self.__get_best_match_from_photos(ordered_featureKeys[-1][1], ignore_r_i)
                H_r, status = self.__get_homography(best_match_r, ordered_featureKeys[-1][0],
                                                    best_i_r, self.__reprojThresh)
                # not really from the right match
                if H_r[0, 2] <= 0:
                    ignore_r_i.append(best_i_r)
                    best_i_r = -1  # continue to search

            # we found a possible match from the right and possible match from the left.
            # we shall only choose to keep the better match, to minimize errors (this makes it
            # possible for a situation where image 5 was match to 123 from the left and 4 was
            # matched on the right. we hope that the 4 will have more matches, and so 5 will get
            # stitched correctly to 1234 next iteration
            if best_i_l != -1 and len(best_match_l) > len(best_match_r):
                ordered_photos = [self.__photos[best_i_l]] + ordered_photos
                ordered_featureKeys = [self.__featuresKeys[best_i_l]] + ordered_featureKeys
                homographies.append(H_l)
                del self.__photos[best_i_l]
                del self.__featuresKeys[best_i_l]
            elif best_i_r != -1:
                ordered_photos.append(self.__photos[best_i_r])
                ordered_featureKeys.append(self.__featuresKeys[best_i_r])
                homographies.append(H_r)
                del self.__photos[best_i_r]
                del self.__featuresKeys[best_i_r]
            else:
                raise NotImplementedError("Can't connect all photos")
                # todo: then what? continue to the other photos?
        # all photos possible were connected
        self.__photos = ordered_photos
        self.__featuresKeys = ordered_featureKeys
        return homographies
    def __warp_and_stitch(self, H, best_index, panorama):
        # check if warped photo is from the left or above the panorama
        p1 = H @ [0, 0, 1]
        p2 = H @ [0, self.__photos[best_index].shape[0] - 1, 1]
        p3 = H @ [self.__photos[best_index].shape[1] - 1, self.__photos[best_index].shape[0] - 1, 1]
        p4 = H @ [self.__photos[best_index].shape[1] - 1, 0, 1]
        p1 = p1 * (1.0 / p1[2])
        p2 = p2 * (1.0 / p2[2])
        p3 = p3 * (1.0 / p3[2])
        p4 = p4 * (1.0 / p4[2])
        # for the panorama's coordinations system - position of image corners after transform
        xmin = min(p1[0], p2[0], p3[0], p4[0], 0)
        ymin = min(p1[1], p2[1], p3[1], p4[1], 0)
        xmax = max(p1[0], p2[0], p3[0], p4[0], panorama.shape[1])
        ymax = max(p1[1], p2[1], p3[1], p4[1], panorama.shape[0])

        width = int(np.ceil(xmax - xmin))
        height = int(np.ceil(ymax - ymin))

        # if one of them is negative - the photo will be cut !
        if xmin < 0 or ymin < 0:
            # needs to translate both the photo and the panorama, so nothing will be cropped
            T = np.array([[1, 0, -xmin],
                          [0, 1, -ymin],
                          [0, 0, 1]])
            TH = T @ H
            result = cv2.warpPerspective(self.__photos[best_index].astype(np.float64), TH, (width, height), borderValue=(np.nan, np.nan, np.nan))
            t_panorama = cv2.warpPerspective(panorama.astype(np.float64), T, (width, height), borderValue=(np.nan, np.nan, np.nan))
            # find the new position of the panorama
            panorama_pos = ~np.isnan(t_panorama[:,:,0])
            result[panorama_pos] = t_panorama.astype(np.uint8)[panorama_pos]
        else:
            result = cv2.warpPerspective(self.__photos[best_index].astype(np.float64), H, (width, height), borderValue=(np.nan, np.nan, np.nan))
            panorama_pos = ~np.isnan(panorama[:, :, 0])
            result[panorama_pos] = panorama.astype(np.uint8)[panorama_pos]

        # transform the panorama image to grayscale and threshold it
        gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds contours from the binary image
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # get the maximum contour area
        c = max(cnts, key=cv2.contourArea)

        # get a bbox from the contour area
        (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
        result = result[y:y + h, x:x + w]
        return result

    def __get_homography(self, matches, panorama_kps, best_index, reprojThresh):  # B is panorama
        kps_new = np.float32([kp.pt for kp in self.__featuresKeys[best_index][0]])
        kps_panorama = np.float32([kp.pt for kp in panorama_kps])

        if len(matches) > 4:
            # construct the two sets of points
            pts_new = np.float32([kps_new[m.queryIdx] for m in matches])
            pts_panorama = np.float32([kps_panorama[m.trainIdx] for m in matches])

            # estimate the homography between the sets of points
            return cv2.findHomography(pts_new, pts_panorama, cv2.RANSAC, reprojThresh)
        else:
            return None

    def __get_best_match_from_photos(self, to_match_features, ignore_idxs=[]):
        best_matches = []
        best_index = None
        for i in range(len(self.__featuresKeys)):
            if i in ignore_idxs:
                continue
            matches = self.__matchKeyPoints(self.__featuresKeys[i][1], to_match_features)
            if len(matches) > len(best_matches):
                best_index = i
                best_matches = matches
        return best_index, best_matches

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

    def __matchKeyPoints(self, featuresA, featuresB):
        crossCheck = ( self.__matcher_type == MatcherType.BF )
        bf = self.__createMatcher(crossCheck)

        if self.__matcher_type == MatcherType.BF:
            best_matches = bf.match(featuresA, featuresB)
            return sorted(best_matches, key = lambda x:x.distance)

        elif self.__matcher_type == MatcherType.KNN:
            rawMatches = bf.knnMatch(featuresA, featuresB, 2)
            matches = []
            for m, n in rawMatches:
                # ensure the distance is within a certain ratio of each
                # other (i.e. Lowe's ratio test)
                if m.distance < n.distance * self.__ratio:
                    matches.append(m)
            return matches
        else:
            raise ( "shouldn't get here!" )
