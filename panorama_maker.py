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

'''
Methods for Keypoints and Descriptors extraction
'''
class DescriptorType(Enum):
    ORB = 1
    BRISK = 2
    SURF = 3
    SIFT = 4
    AKAZE = 5


'''
Having features from two images - this algorithms match those features 
'''
class MatcherType(Enum):
    BF = 1
    KNN = 2

'''
The main class
counter - will contain the number of images (tot) that's going to be stitched
check_first - first photo we're going to work on
ratio (KNN) - For each pair of features, if the distance between them is within a certain ratio
        we keep it, otherwise, we throw it away (defined by lowest)
reprojThresh - parameter for RANSAC for homography computation        
'''
class PanoramaMaker:
    def __init__(self, descriptor_type=DescriptorType.ORB, matcher_type=MatcherType.BF,
                 ratio=0.75, reprojThresh=4):
        self.__counter = 0  # todo do I need that counter? maybe for later use
        self.__photos = []
        self.__featuresKeys = []
        self.__check_first = 0
        self.__descriptor_type = descriptor_type
        self.__descriptor_func = self.__choose_descriptor_type()  # used in detectAndDescribe
        self.__matcher_type = matcher_type
        self.__ratio = ratio
        self.__reprojThresh = reprojThresh
        self.__homographies = []

    def add_photo(self, image):
        """images pre-processing

        converts images to grayscale, then extract Key points and descriptors ('features')
        We fill featuresKeys[] to hold that data (as a tuple)
        Also find the index to the photo which holds the most features (check_first)

        """
        # read image
        self.__photos.append(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # for data extraction
        key_points, photo_features = self.__detect_and_describe(image_gray)  # getting information
        # adding the photo's information to the list as a tuple
        self.__featuresKeys.append((key_points, photo_features))
        if self.__check_first < len(photo_features):  # for later use - who's will be first
            self.__check_first = self.__counter
        self.__counter += 1
        
    '''
    The Main func - creates the panorama
    '''
    def create_panorama(self):
        self.__homographies = self.__reorder()
        f = self.__estimate_focal_length()

        print("[+] Stitching panorama...")
        panorama = self.__photos[0]
        # panorama_kps, panorama_features = self.__featuresKeys[0]
        # del self.__photos[0]
        # del self.__featuresKeys[0]
        # self.__counter -= 1

        # loop for all other photos
        for i in range(1, len(self.__photos) + 1):
            # The stitched photo is the next one
            panorama = self.__warp_and_stitch(self.__homographies[i], i, panorama)
            # delete used photo
            # del self.__photos[best_index]
            # del self.__featuresKeys[best_index]
            # self.__counter -= 1
            # panorama_gray = cv2.cvtColor(panorama.astype(np.uint8), cv2.COLOR_RGB2GRAY)  # for data extraction
            # panorama_kps, panorama_features = self.__detectAndDescribe(panorama_gray)  # getting information on new panorama

        return panorama.astype(np.uint8)

    def __reorder(self):
        print('[+] Getting photo order...')
        ordered_photos = [self.__photos[self.__check_first]]
        ordered_featureKeys = [self.__featuresKeys[self.__check_first]]
        homographies = []
        del self.__photos[self.__check_first]
        del self.__featuresKeys[self.__check_first]

        while len(self.__photos):
            print("[*] Connecting %d photos..." % len(self.__photos), end='\r')
            best_i_r = -1
            best_i_l = -1
            H_r = None
            H_l = None
            best_match_r = None
            best_match_l = None
            # who is the best from the left
            ignore_l_i = []
            while best_i_l == -1 and len(ignore_l_i) < len(self.__photos):
                best_i_l, best_match_l = self.__get_best_match_from_photos(
                    ordered_featureKeys[0][1], ignore_l_i)
                H_l, status = self.__get_homography(best_match_l, ordered_featureKeys[0][0],
                                                    best_i_l, self.__reprojThresh)
                # not really from the left match
                if H_l[0, 2] >= 0:
                    ignore_l_i.append(best_i_l)
                    best_i_l = -1  # continue to search

            # who is the best from the right
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

    def __warp_and_stitch(self, H, dst_index, panorama):
        # check if warped photo is from the left or above the panorama
        p1 = H @ [0, 0, 1]
        p2 = H @ [0, self.__photos[dst_index].shape[0] - 1, 1]
        p3 = H @ [self.__photos[dst_index].shape[1] - 1, self.__photos[dst_index].shape[0] - 1, 1]
        p4 = H @ [self.__photos[dst_index].shape[1] - 1, 0, 1]
        p1 = p1 * (1.0 / p1[2])
        p2 = p2 * (1.0 / p2[2])
        p3 = p3 * (1.0 / p3[2])
        p4 = p4 * (1.0 / p4[2])
        # for the panorama's coordinate system - position of image corners after transform
        x_min = min(p1[0], p2[0], p3[0], p4[0], 0)
        y_min = min(p1[1], p2[1], p3[1], p4[1], 0)
        x_max = max(p1[0], p2[0], p3[0], p4[0], panorama.shape[1])
        y_max = max(p1[1], p2[1], p3[1], p4[1], panorama.shape[0])

        width = int(np.ceil(x_max - x_min))
        height = int(np.ceil(y_max - y_min))

        # if one of them is negative - the photo will be cut !
        if x_min < 0 or y_min < 0:
            # needs to translate both the photo and the panorama, so nothing will be cropped
            T = np.array([[1, 0, -x_min],
                          [0, 1, -y_min],
                          [0, 0, 1]])
            TH = T @ H
            result = cv2.warpPerspective(self.__photos[dst_index].astype(np.float64), TH,
                                         (width, height), borderValue=(np.nan, np.nan, np.nan))
            t_panorama = cv2.warpPerspective(panorama.astype(np.float64), T,
                                             (width, height), borderValue=(np.nan, np.nan, np.nan))
            # find the new position of the panorama
            panorama_pos = ~np.isnan(t_panorama[:, :, 0])
            result[panorama_pos] = t_panorama.astype(np.uint8)[panorama_pos]
        else:
            result = cv2.warpPerspective(self.__photos[dst_index].astype(np.float64), H,
                                         (width, height), borderValue=(np.nan, np.nan, np.nan))
            panorama_pos = ~np.isnan(panorama[:, :, 0])
            result[panorama_pos] = panorama.astype(np.uint8)[panorama_pos]

        # transform the panorama image to grayscale and threshold it
        gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds contours from the binary image
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # get the maximum contour area
        c = max(contours, key=cv2.contourArea)

        # get a bbox from the contour area
        (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
        result = result[y:y + h, x:x + w]
        return result

    def __get_homography(self, matches, panorama_kps, best_index, reproj_thresh):  # B is panorama
        kps_new = np.float32([kp.pt for kp in self.__featuresKeys[best_index][0]])
        kps_panorama = np.float32([kp.pt for kp in panorama_kps])

        if len(matches) > 4:
            # construct the two sets of points
            pts_new = np.float32([kps_new[m.queryIdx] for m in matches])
            pts_panorama = np.float32([kps_panorama[m.trainIdx] for m in matches])

            # estimate the homography between the sets of points
            return cv2.findHomography(pts_new, pts_panorama, cv2.RANSAC, reproj_thresh)
        else:
            return None

    def __get_best_match_from_photos(self, to_match_features, ignore_idxs):
        best_matches = []
        best_index = None
        for i in range(len(self.__featuresKeys)):
            if i in ignore_idxs:
                continue
            matches = self.__match_keypoints(self.__featuresKeys[i][1], to_match_features)
            if len(matches) > len(best_matches):
                best_index = i
                best_matches = matches
        return best_index, best_matches

    def __estimate_focal_length(self):
        focals = []
        for H in self.__homographies:
            m0 = H[0, 0]
            m1 = H[0, 1]
            m2 = H[0, 2]
            m3 = H[1, 0]
            m4 = H[1, 1]
            m5 = H[1, 2]
            m6 = H[2, 0]
            m7 = H[2, 1]

            width = self.__photos[0].shape[1]
            height = self.__photos[0].shape[0]
            cx = width / 2
            cy = height / 2

            d0a, d0b, d1a, d1b = 0, 0, 0, 0
            f0a, f0b, f1a, f1b = 0, 0, 0, 0

            try:
                d0a = (m0 - m6 * cx) ** 2 + (m1 - m7 * cx) ** 2 - \
                      (m3 - cy * m6) ** 2 - (m4 - cy * m7) ** 2
                f0a = np.sqrt(((m5 + cx * m3 + cy * m4 - cy * (1 + cx * m6 + cy * m7)) ** 2 - (
                                m2 + cx * m0 + cy * m1 - cx * (1 + cx * m6 + cy * m7)) ** 2)
                              / d0a)
            except RuntimeWarning:
                pass
            try:
                d0b = (m0 - m6 * cx) * (m3 - cy * m6) + (m1 - m7 * cx) * (m4 - cy * m7)
                f0b = np.sqrt(-1 * ((m5 + cx * m3 + cy * m4 - cy * (1 + cx * m6 + cy * m7)) *
                                    (m2 + cx * m0 + cy * m1 - cx * (1 + cx * m6 + cy * m7)))
                              / d0b)
            except RuntimeWarning:
                pass
            try:
                d1a = m7 ** 2 - m6 ** 2
                f1a = np.sqrt(((m0 - m6 * cx) ** 2 - (m1 - m7 * cx) ** 2 +
                               (m3 - cy * m6) ** 2 - (m4 - cy * m7) ** 2) / d1a)
            except RuntimeWarning:
                pass
            try:
                d1b = m6 * m7
                f1b = np.sqrt(-1 * ((m0 - m6 * cx) * (m1 - m7 * cx) +
                                    (m3 - m6 * cy) * (m4 - cy * m7)) / d1b)
            except RuntimeWarning:
                pass

            # choose best focal lengths
            f1 = f1a or f1b
            if f1a and f1b:
                f1 = f1a if np.abs(d1a) > np.abs(d1b) else f1b
            f0 = f0a or f0b
            if f0a and f0b:
                f0 = f0a if np.abs(d0a) > np.abs(d0b) else f0b
            focals.append(np.sqrt(f0*f1))

        focals = np.array(focals)[np.array(focals) != 0]
        if len(focals):
            return np.median(focals)
        else:
            raise Exception("[X] Couldn't estimate focals")

    '''
    compute Key points and features descriptors
    '''
    def __detect_and_describe(self, image):
        descriptor = self.__descriptor_func()
        return descriptor.detectAndCompute(image, None)

    '''
    creates a BruteForce Matcher using OpenCV
    for string-based descriptors, and Hamming distance for binary descriptors
    cross_check - for a pair of features to considered valid, f1 needs to match f2 and vice versa
    '''
    def __create_matcher(self, cross_check):
        if self.__descriptor_type == DescriptorType.SIFT or\
                self.__descriptor_type == DescriptorType.SURF:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
        elif self.__descriptor_type == DescriptorType.ORB or self.__descriptor_type ==\
                DescriptorType.BRISK or self.__descriptor_type == DescriptorType.AKAZE:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        else:
            # We dealt with that previously, ORB is default
            raise Exception("should not get here")
        return bf

    '''
    The matcher algorithm will give us the best (more similar) set of features from both images
    '''
    def __match_keypoints(self, features_a, features_b):
        cross_check = (self.__matcher_type == MatcherType.BF)
        bf = self.__create_matcher(cross_check)

        if self.__matcher_type == MatcherType.BF:
            best_matches = bf.match(features_a, features_b)
            return sorted(best_matches, key=lambda x: x.distance)

        elif self.__matcher_type == MatcherType.KNN:
            raw_matches = bf.knnMatch(features_a, features_b, 2)
            matches = []
            for m, n in raw_matches:
                # ensure the distance is within a certain ratio of each
                # other (i.e. Lowe's ratio test)
                if m.distance < n.distance * self.__ratio:
                    matches.append(m)
            return matches
        else:
            raise Exception("shouldn't get here!")

    '''
    The user may choose his algorithm for features extraction (5 types)
    '''
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


'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def __warp_cylindrical_to_cartesian(img1, focal_length):

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    # np.zeros_like : Return an array of zeros with the same shape and type as a given array
    cyl_mask = np.zeros_like(img1)
    cyl_h, cyl_w = cyl.shape                  #check
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    x, y = np.meshgrid(np.arange(0,cyl_w), np.arange(0,cyl_h))
    theta = (x - x_c) / focal_length
    h = (y - y_c) / focal_length
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    flat_h = np.reshape(h, (1,-1))
    flat_sin_theta = np.reshape(sin_theta, (1, -1))
    flat_cos_theta = np.reshape(cos_theta, (1, -1))
    cartesian = np.stack((flat_sin_theta, flat_h, flat_cos_theta), axis=0)
    K = np.array([[focal_length, 0, cyl_w / 2], [0, focal_length, cyl_h / 2], [0, 0, 1]])
    # calibration matrix
    cylX = np.dot(K, X)
    x_im = cylX[0] / cylX[2]
    y_im = cylX[1] / cylX[2]
    valid_x_im = x_im[(x_im > 0) & (x_im < im_w)]
    valid_y_im = y_im[(y_im > 0) & (y_im < im_h)]
    xy = np.stack((x_im.reshape(img1.shape), y_im.reshape(img1.shape)), axis=2)
    idx_valid_x, idx_valid_y = ( (xy[:, :, 0] > 0) & (xy[:, :, 0] < im_w) &
                                 (xy[:, :, 1] > 0) & (xy[:, :, 1] < im_h) ).nonzero()
    cyl[idx_valid_x, idx_valid_y] = img1[int(valid_y_im), int(valid_x_im)]
    cyl_mask[idx_valid_x, idx_valid_y] = 255
    return cyl, cyl_mask
