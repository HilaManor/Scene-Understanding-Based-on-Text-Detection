import matplotlib.pyplot as plt

import matplotlib
import cv2
import matplotlib.pyplot as plt
from pylab import *
import os
from PIL import Image


def convert_hsv_histogram(street_sign):
    bins = 16
    range = (0, 255)
    rgb_img = cv2.imread(street_sign, cv2.IMREAD_UNCHANGED)
    mask_img = rgb_img[:, :, -1]
    _, ret = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    mask = ret.astype(np.bool)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    hue_img = hsv_img[:, :, 0]
    saturation_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]
    plt.subplot(2, 2, 1)
    title('Histogram of the Hue channel')
    hist(hue_img[mask], bins=bins, range=range)
    plt.subplot(2, 2, 2)
    title('Histogram of the saturation channel')
    hist(saturation_img[mask], bins=bins, range=range)
    plt.subplot(2, 2, 3)
    title('Histogram of the Value channel')
    hist(value_img[mask], bins=bins, range=range)
    plt.subplot(2, 2, 4)
    imshow(rgb_img)
    title('RGB_img')
    plt.savefig(os.path.join(r"C:\Users\user\Desktop\output_street_signs_fixed_range",
                             os.path.basename(street_sign)))
    plt.show()




    #fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))
    #ax0.hist(hue_img.ravel(), 512)
    #ax0.set_title("Histogram of the Hue channel")
    #ax1.imshow(hue_img)
    #ax1.set_title("Hue image")
    #ax1.axis('off')
    #fig.tight_layout()
    #plt.show()

    pass

if __name__ == '__main__':
    plt.close('all')
    dir = r"C:\Users\user\Desktop\FINALS"
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        convert_hsv_histogram(path)