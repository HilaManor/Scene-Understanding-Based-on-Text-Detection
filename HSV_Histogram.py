import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hsv
import matplotlib
import cv2
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image


def convert_hsv_histogram(street_sign):
    rgb_img = cv2.imread(street_sign, cv2.IMREAD_UNCHANGED)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    hue_img = hsv_img[:, :, 0]
    saturation_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]
    plt.subplot(2, 2, 1)
    title('Histogram of the Hue channel')
    hist(hue_img.ravel(), 512)
    plt.subplot(2, 2, 2)
    title('Histogram of the saturation channel')
    hist(saturation_img.ravel(), 512)
    plt.subplot(2, 2, 3)
    title('Histogram of the Value channel')
    hist(value_img.ravel(), 512)
    plt.subplot(2, 2, 4)
    imshow(hsv_img)
    title('hsv_img')
    show()





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
    convert_hsv_histogram(r"C:\Users\user\Desktop\check_7.png")