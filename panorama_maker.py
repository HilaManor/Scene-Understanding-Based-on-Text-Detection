"""This module handles the 1st algorithm stage: Create a panoramic photo

The creation is robust to a full or partial field of view and unordered input.
Options exist to change the descriptors and matchers.
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import numpy as np
import imutils
from enum import Enum
cv2.ocl.setUseOpenCL(False)


class DescriptorType(Enum):
    """ Available descriptors enum"""
    ORB = 1
    BRISK = 2
    SURF = 3
    SIFT = 4
    AKAZE = 5


class MatcherType(Enum):
    """Available Matcher options enum"""
    BF = 1
    KNN = 2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~


class PanoramaMaker:
    """Responsible for creating a panorama from a full/partial field of view with unsorted input.

    Public Functions:
        add_photo - adds a photo to the instance to operate on
        create_panorama - Create a panorama from the photos added until being called

    Variables:
        descriptor_type - DescriptorType to match features with.
        matcher_type - MatcherType to match features with
        ratio(KNN) - accepted distance ratio between matches
        reprojection_thresh - RANSAC threshold

    Private Variables:
        __counter - amount of photos added to the instance
        __photos - inserted raw photos
        __featuresKeys - list of tuples, each tuple has features list and key-points list for its
            corresponding photo
        __check_first - First photo to start reordering from (middle of the panorama),
            will be chosen as the photo with the most features detected
        __homographies - homographies calculated between consecutive pairs of photos
    """
    def __init__(self, descriptor_type=DescriptorType.ORB, matcher_type=MatcherType.BF,
                 ratio=0.75, reprojection_thresh=4):
        self.__counter = 0
        self.__photos = []
        self.__featuresKeys = []
        self.__check_first = 0
        self.__descriptor_type = descriptor_type
        self.__descriptor_func = self.__choose_descriptor_type()  # used in detectAndDescribe
        self.__matcher_type = matcher_type
        self.__ratio = ratio
        self.__reprojThresh = reprojection_thresh
        self.__homographies = []

    def add_photo(self, image):
        """Add an image to be processed

        extract features from the added image.
        update __check_first to hold the index to the image with the most detections

        :param image: OpenCV image
        :return: None
        """

        self.__photos.append(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # for data extraction
        key_points, photo_features = self.__detect_and_describe(image_gray)  # getting information
        # adding the photo's information to the list as a tuple
        self.__featuresKeys.append((key_points, photo_features))
        if self.__check_first < len(photo_features):  # for later use - who's will be first
            self.__check_first = self.__counter
        self.__counter += 1

    def create_panorama(self, dont_reorder):
        """Create the panorama from the images that were added until call

        :param dont_reorder: Boolean for weather to treat the images as order, or to search the
            true order. Will reduce runtime immensely.
        :return: panorama image as uint8
        """

        self.__homographies = self.__reorder(dont_reorder)
        f = self.__estimate_focal_length()

        print("[+] warping photos to cylindrical format...")
        affines, cyl_featureKeys, cyl_photos = self.__make_everything_cylindrical(f)

        print("[+] Stitching panorama...")
        # we stitch from the middle to avoid as much rotation as we can
        middle_idx = (self.__counter//2)
        panorama = cyl_photos[middle_idx]  # middle
        A = np.eye(2, 3)
        y_min = 0
        x_min = 0
        for idx in range(middle_idx+1, len(self.__photos)):  # middle to the right
            A = _multiply_affine(A, affines[idx-1])
            x_min, y_min, panorama = self.__stitch_cylindrical(panorama, cyl_photos[idx],
                                                               A, y_min, x_min)

        A = np.eye(2, 3)
        for idx in range(middle_idx-1, -1, -1):  # middle to the left, last performance is idx = 1
            padded_inved_mat = np.linalg.inv(np.vstack((affines[idx], [0, 0, 1])))
            inversed_affine = padded_inved_mat[:2, ]
            A = _multiply_affine(A, inversed_affine)
            x_min, y_min, panorama = self.__stitch_cylindrical(panorama, cyl_photos[idx],
                                                               A, y_min, x_min)

        panorama = self.__crop_boundaries(panorama)
        return panorama.astype(np.uint8)

    def __reorder(self, dont_reorder):
        """Find the homographies between each pair of photos, Reorder if needed.

        :param dont_reorder: Boolean for weather to treat the images as order, or to search the
            true order. Will reduce runtime immensely.
        :return: homographies between each pair of consecutive photos in the __photos list
        """

        print('[+] Getting photo order...')
        first_photo = 0 if dont_reorder else self.__check_first
        ordered_photos = [self.__photos[first_photo]]
        ordered_featureKeys = [self.__featuresKeys[first_photo]]
        homographies = []
        del self.__photos[first_photo]
        del self.__featuresKeys[first_photo]

        while len(self.__photos):
            print("[*] Connecting %d photos..." % len(self.__photos), end='\r')
            if dont_reorder:  # just compute homographies
                matches = self.__match_keypoints(self.__featuresKeys[0][1],
                                                 ordered_featureKeys[-1][1])
                H_r, status = self.__get_homography(matches, ordered_featureKeys[-1][0],
                                                    self.__featuresKeys[0][0],
                                                    self.__reprojThresh)

                ordered_photos.append(self.__photos[0])
                ordered_featureKeys.append(self.__featuresKeys[0])
                homographies.append(H_r)
                del self.__photos[0]
                del self.__featuresKeys[0]
            else:
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
                                                        self.__featuresKeys[best_i_l][0],
                                                        self.__reprojThresh)
                    # not really from the left match
                    if H_l[0, 2] >= 0:
                        ignore_l_i.append(best_i_l)
                        best_i_l = -1  # continue to search

                # who is the best from the right
                ignore_r_i = []
                while best_i_r == -1 and len(ignore_r_i) < len(self.__photos):
                    best_i_r, best_match_r = self.__get_best_match_from_photos(
                        ordered_featureKeys[-1][1], ignore_r_i)
                    H_r, status = self.__get_homography(best_match_r, ordered_featureKeys[-1][0],
                                                        self.__featuresKeys[best_i_r][0],
                                                        self.__reprojThresh)
                    # not really from the right match
                    if H_r[0, 2] <= 0:
                        ignore_r_i.append(best_i_r)
                        best_i_r = -1  # continue to search

                # we found a possible match from the right and possible match from the left.
                # we shall only choose to keep the better match, to minimize errors (this makes it
                # possible for a situation where image 5 was match to 123 from the left and 4 was
                # matched on the right. we hope that the 4 will have more matches, and so 5 will
                # get stitched correctly to 1234 next iteration
                if best_i_l != -1 and len(best_match_l) >= len(best_match_r):
                    ordered_photos = [self.__photos[best_i_l]] + ordered_photos
                    ordered_featureKeys = [self.__featuresKeys[best_i_l]] + ordered_featureKeys
                    homographies.append(np.linalg.inv(H_l))
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
        # all photos possible were connected
        self.__photos = ordered_photos
        self.__featuresKeys = ordered_featureKeys
        return homographies

    def __make_everything_cylindrical(self, f):
        """Apply Inverse cylindrical warp to all the photos

        :param f: average focal length for the images
        :return: tuple of:
                    Affine transforms for each pair of photos
                    tuple of feature list and array list for each of the cylindrical photos
                    the cylindrical-warped photos
        """

        cyl_photo, mask = _warp_cylindrical_to_cartesian(self.__photos[0], f)
        cyl_photos = [_fill_with_nan_mask(cyl_photo, mask)]
        image_gray = cv2.cvtColor(cyl_photos[0].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        key_points, photo_features = self.__detect_and_describe(image_gray)
        cyl_featureKeys = [(key_points, photo_features)]
        affines = []

        for i in range(1, len(self.__photos)):
            cyl_photo, mask = _warp_cylindrical_to_cartesian(self.__photos[i], f)
            cyl_photos.append(_fill_with_nan_mask(cyl_photo, mask))
            image_gray = cv2.cvtColor(cyl_photo.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            key_points, photo_features = self.__detect_and_describe(image_gray)
            cyl_featureKeys.append((key_points, photo_features))
            matches = self.__match_keypoints(cyl_featureKeys[i][1], cyl_featureKeys[i-1][1])
            A, status = self.__get_homography(matches, cyl_featureKeys[i-1][0],
                                              cyl_featureKeys[i][0], self.__reprojThresh,
                                              is_affine=True)
            affines.append(A)

        return affines, cyl_featureKeys, cyl_photos

    def __stitch_cylindrical(self, base_photo, add_photo, A, y_min_last=0, x_min_last=0):
        """Stitch a cylindrical photo to another photo (the current panorama)

        We compensate for x,y that exit the image boundaries (negative coords),
        but for that we need to keep track of the translation of the image origin,
        which isn't described by the affine matrices.

        :param base_photo: base photo to stitch to (the current panorama)
        :param add_photo: the cylindrical photo to stitch
        :param A: affine transform from the photo to the base photo
        :param y_min_last: last y amount that was compensated for
        :param x_min_last: last x amount that was compenstaed for
        :return: tuple of
                    next x amount to compensate for,
                    next y amount to compensate for,
                    stitched photo
        """

        # Find the new photo's corners' coords in the pano frame of reference (coordinate system)
        p1 = A @ [0, 0, 1]
        p2 = A @ [0, add_photo.shape[0] - 1, 1]
        p3 = A @ [add_photo.shape[1] - 1, add_photo.shape[0] - 1, 1]
        p4 = A @ [add_photo.shape[1] - 1, 0, 1]

        # find total image corners position, in panorama frame of reference (coordinate system)
        x_min = min(p1[0], p2[0], p3[0], p4[0], 0)
        y_min = min(p1[1], p2[1], p3[1], p4[1], 0)
        x_max = max(p1[0], p2[0], p3[0], p4[0], base_photo.shape[1])
        y_max = max(p1[1], p2[1], p3[1], p4[1], base_photo.shape[0])

        width = int(np.ceil(x_max - x_min))
        height = int(np.ceil(y_max - y_min))

        # To compensate for negative boundaries, move the center point
        T_cut = np.eye(2, 3)
        T_cut_with_last = np.array([[1, 0, -x_min_last],
                                    [0, 1, -y_min_last]])
        y_min_next = y_min_last
        # Check if we need to move even more from the last image (Y axis)
        if y_min - y_min_last < 0:
            T_cut[1, 2] = -(y_min - y_min_last)
            T_cut_with_last[1, 2] = -y_min
            y_min_next = y_min

        x_min_next = x_min_last

        # Check if we need to move even more from the last image (X axis)
        if x_min - x_min_last < 0:
            T_cut[0, 2] = -(x_min - x_min_last)
            T_cut_with_last[0, 2] = -x_min
            x_min_next = x_min

        # add translation to affine matrix)
        TA = _multiply_affine(T_cut_with_last, A)

        # needs to translate both the photo and the panorama, so nothing will be cropped
        result = cv2.warpAffine(add_photo, TA, (width, height),
                                borderValue=(np.nan, np.nan, np.nan), flags=cv2.INTER_CUBIC)
        t_base_photo = cv2.warpAffine(base_photo.astype(np.float64), T_cut, (width, height),
                                      borderValue=(np.nan, np.nan, np.nan), flags=cv2.INTER_CUBIC)

        # find the new position of the base_photo
        base_photo_pos = ~np.isnan(t_base_photo[:, :, 0])

        # clamp out-of-range values
        result[result[:, :, :] < 0] = 0
        result[result[:, :, :] > 255] = 255
        t_base_photo[t_base_photo[:, :, :] < 0] = 0
        t_base_photo[t_base_photo[:, :, :] > 255] = 255

        result[base_photo_pos] = t_base_photo.astype(np.uint8)[base_photo_pos]
        return x_min_next, y_min_next, result

    def __crop_boundaries(self, crop):
        """Crop excess (black) boundaries of image

        :param crop: image to crop
        :return: cropped image
        """

        # transform the panorama image to grayscale and threshold it
        gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds contours from the binary image
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # get the maximum contour area
        c = max(contours, key=cv2.contourArea)

        # get a bbox from the contour area
        (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
        return crop[y:y + h, x:x + w]

    def __get_homography(self, matches, dst_kps, src_kps, reproj_thresh, is_affine=False):
        """Find homography from one photo to another

        :param matches: matches between both photos
        :param dst_kps: Destination frame of reference key-points
        :param src_kps: Source frame of reference key-points
        :param reproj_thresh: RANSAC threshold
        :param is_affine: Boolean to find an affine transforamtion instead of homography
        :return: (Homography/Affine matrix, status)
        """

        kps_new = np.float32([kp.pt for kp in src_kps])
        kps_panorama = np.float32([kp.pt for kp in dst_kps])

        if len(matches) > 4:
            # construct the two sets of points
            pts_new = np.float32([kps_new[m.queryIdx] for m in matches])
            pts_panorama = np.float32([kps_panorama[m.trainIdx] for m in matches])

            # estimate the homography between the sets of points
            if is_affine:
                return cv2.estimateAffine2D(pts_new, pts_panorama, cv2.RANSAC, reproj_thresh)
            else:
                return cv2.findHomography(pts_new, pts_panorama, cv2.RANSAC, reproj_thresh)
        else:
            return None

    def __get_best_match_from_photos(self, to_match_features, ignore_idxs):
        """Find the best matching photo from current photos toe given features

        :param to_match_features: features to match to
        :param ignore_idxs: which photos to ignore
        :return: best photo index, and the matched matches
        """

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
        """Estimate the average focal length for all the photos

        :return: estimated average focal length
        """

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
            f1 = f1a
            if not np.isnan(f1a) and not np.isnan(f1b):
                f1 = f1a if np.abs(d1a) > np.abs(d1b) else f1b
            elif np.isnan(f1a):
                f1 = f1b
            f0 = f0a
            if not np.isnan(f0a) and not np.isnan(f0b):
                f0 = f0a if np.abs(d0a) > np.abs(d0b) else f0b
            elif np.isnan(f0a):
                f0 = f0b

            if np.isnan(f0) and np.isnan(f1):
                f1 = f0 = 0
            elif np.isnan(f1):
                f1 = f0
            elif np.isnan(f0):
                f0 = f1
            focals.append(np.sqrt(f0*f1))

        focals = np.array(focals)[np.array(focals) != 0]
        if len(focals):
            return np.median(focals)
        else:
            raise Exception("[X] Couldn't estimate focals")

    def __detect_and_describe(self, image):
        """Compute Key-points and features descriptors

        :param image: image to detect
        :return: kepoints and features
        """

        descriptor = self.__descriptor_func()
        return descriptor.detectAndCompute(image, None)

    def __create_matcher(self, cross_check):
        """Creates a matcher for the chosen (at init) matcher.

        :param cross_check: Do matches need to be matched from both images to one another - bool
        :return: matcher
        """

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

    def __match_keypoints(self, features_a, features_b):
        """Match best features from 2 set of images

        :param features_a: features of image 1
        :param features_b: features of image 2
        :return: matched features
        """

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

    def __choose_descriptor_type(self):
        """ Descriptor creation based on chosen (at init) descriptor

        :return:
        """

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


def _warp_cylindrical_to_cartesian(img1, focal_length):
    """Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)

    Returns: (image, mask)
    Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
    Masks are useful for stitching

    Usage example:

        im = cv2.imread("myimage.jpg",0) #grayscale
        h,w = im.shape
        f = 700
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
        imcyl = cylindricalWarpImage(im, K)
    """

    im_h, im_w, _ = img1.shape
    # go inverse from cylindrical coord to the image (this way there are no gaps)
    cyl = np.zeros_like(img1)

    # np.zeros_like : Return an array of zeros with the same shape and type as a given array
    cyl_mask = np.zeros_like(img1, dtype=np.bool)
    cyl_h, cyl_w, _ = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    x, y = np.meshgrid(np.arange(0, cyl_w), np.arange(0, cyl_h))
    theta = (x - x_c) / focal_length
    h = (y - y_c) / focal_length
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    flat_h = np.reshape(h, (1, -1))
    flat_sin_theta = np.reshape(sin_theta, (1, -1))
    flat_cos_theta = np.reshape(cos_theta, (1, -1))
    cartesian_coords = np.vstack((flat_sin_theta, flat_h, flat_cos_theta))
    K = np.array([[focal_length, 0, cyl_w / 2], [0, focal_length, cyl_h / 2], [0, 0, 1]])
    # calibration matrix
    cyl_coords = np.dot(K, cartesian_coords)
    x_im = cyl_coords[0] / cyl_coords[2]
    y_im = cyl_coords[1] / cyl_coords[2]
    xy = np.dstack((x_im.reshape(im_h, im_w), y_im.reshape(im_h, im_w)))
    idx_valid_x, idx_valid_y = ((xy[:, :, 0] > 0) & (xy[:, :, 0] < im_w) &
                                (xy[:, :, 1] > 0) & (xy[:, :, 1] < im_h)).nonzero()
    valid_x_im = xy[idx_valid_x, idx_valid_y][:, 0]
    valid_y_im = xy[idx_valid_x, idx_valid_y][:, 1]
    cyl[idx_valid_x, idx_valid_y] = img1[valid_y_im.astype(int), valid_x_im.astype(int)]
    cyl_mask[idx_valid_x, idx_valid_y] = True
    return cyl, cyl_mask


def _multiply_affine(A, B):
    """ Multiply affine matrices (2x3 dims)

    :param A: Affine 1
    :param B: Affine 2
    :return: multiplied affine (2x3) matrix
    """

    FA = np.vstack((A, [0, 0, 1]))
    FB = np.vstack((B, [0, 0, 1]))
    FAB = FA @ FB
    return FAB[:-1]


def _fill_with_nan_mask(cyl_photo, mask):
    cyl_photo = cyl_photo.astype(np.float64)
    cyl_photo[~mask] = np.nan
    return cyl_photo
