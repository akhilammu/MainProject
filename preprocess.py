import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import (ball, disk, dilation, binary_erosion,
remove_small_objects, erosion, closing, reconstruction, binary_closing)
from skimage.measure import (label,regionprops, perimeter)
from skimage.morphology import (binary_dilation, binary_opening)
from skimage.filters import (roberts, sobel)
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

img = cv2.imread('/home/mca-pc74/proj/dataset/test.png',0)
img = cv2.getRectSubPix(img, (320, 220), (150, 170))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.show()

(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY |
cv2.THRESH_OTSU)
thresh = 150
im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

plot = True
if plot == True:
    f, plots = plt.subplots(9, 1, figsize=(5, 40))


if plot == True:
    plots[0].axis('off')
    plots[0].imshow(im_bw, cmap=plt.cm.bone)

cleared = clear_border(im_bw)
if plot == True:
    plots[1].axis('off')
    plots[1].imshow(cleared, cmap=plt.cm.bone)

label_image = label(cleared)
if plot == True:
    plots[2].axis('off')
    plots[2].imshow(label_image, cmap=plt.cm.bone)



areas = [r.area for r in regionprops(label_image)]
areas.sort()
if len(areas) > 2:
    for region in regionprops(label_image):
        if region.area < areas[-2]:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 0
binary = label_image > 0


if plot == True:
    plots[3].axis('off')
    plots[3].imshow(binary, cmap=plt.cm.bone)

selem = disk(10)
binary = binary_closing(binary, selem)
if plot == True:
    plots[4].axis('off')
    plots[4].imshow(binary, cmap=plt.cm.bone)

edges = roberts(binary)
binary = ndi.binary_fill_holes(edges)
if plot == True:
    plots[5].axis('off')
    plots[5].imshow(binary, cmap=plt.cm.bone)

get_high_vals = binary == 0
img[get_high_vals] = 0
if plot == True:
    plots[6].axis('off')
    plots[6].imshow(img, cmap=plt.cm.bone)


