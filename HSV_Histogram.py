import matplotlib.pyplot as plt
import cv2
from pylab import *
import os


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
    plt.savefig(os.path.join(r"E:\Hila\Documents\Technion\Semester F\project-A\Data\FINALS\bhistsoutput",
                             os.path.basename(street_sign)))
    plt.show()


if __name__ == '__main__':
    plt.close('all')
    dir = r"E:\Hila\Documents\Technion\Semester F\project-A\Data\FINALS\broadway_hists"
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        convert_hsv_histogram(path)