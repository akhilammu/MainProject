import cv2
import numpy as np 
import pandas as pd 
import skimage, os
from skimage.morphology import (ball, disk, dilation, binary_erosion,
remove_small_objects, erosion, closing, reconstruction, binary_closing)
from skimage.measure import (label,regionprops, perimeter)
from skimage.morphology import (binary_dilation, binary_opening)
from skimage.filters import (roberts, sobel)
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

im = cv2.imread('/home/mca-pc74/proj/dataset/6799964c08ad5ce7740defcd3bd037a6_103_1.png',0)

img = cv2.getRectSubPix(im, (320, 220), (150, 150))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.show()

(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY |
cv2.THRESH_OTSU)
thresh = 136
im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

ret,im_inv = cv2.threshold(im_bw,thresh,255,cv2.THRESH_BINARY_INV)
plot = True
if plot == True:
    f, plots = plt.subplots(10, 1, figsize=(5, 40))


if plot == True:
    plots[0].axis('off')
    plots[0].imshow(im_bw, cmap=plt.cm.bone)
    
if plot == True:
    plots[1].axis('off')
    plots[1].imshow(im_inv, cmap=plt.cm.bone)

cleared = clear_border(im_bw)
if plot == True:
    plots[2].axis('off')
    plots[2].imshow(cleared, cmap=plt.cm.bone)

label_image = label(cleared)
if plot == True:
    plots[3].axis('off')
    plots[3].imshow(label_image, cmap=plt.cm.bone)



areas = [r.area for r in regionprops(label_image)]
areas.sort()
if len(areas) > 2:
    for region in regionprops(label_image):
        if region.area < areas[-2]:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 0
binary = label_image > 0


if plot == True:
    plots[4].axis('off')
    plots[4].imshow(binary, cmap=plt.cm.bone)


selem = disk(10)
binary = binary_closing(binary, selem)
if plot == True:
    plots[5].axis('off')
    plots[5].imshow(binary, cmap=plt.cm.bone)

edges = roberts(binary)
binary = ndi.binary_fill_holes(edges)
if plot == True:
    plots[6].axis('off')
    plots[6].imshow(binary, cmap=plt.cm.bone)

get_high_vals = binary == 0
img[get_high_vals] = 0
if plot == True:
    plots[7].axis('off')
    plots[7].imshow(img, cmap=plt.cm.bone)


image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

numOfContours = len(contours)   #number of contours

area = []
count = 0
for count in range(numOfContours) :
    cv2.drawContours(img, contours, -1, (20,255,60), 1)  #draw contours
 count+=1
if plot == True:
    plots[8].axis('off')
plots[8].imshow(img, cmap=plt.cm.bone)


im_inv = clear_border(im_inv)
image, contours, hierarchy = cv2.findContours(im_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
area = []
count = 0
numOfContours = len(contours)
for count in range(numOfContours) :
    cv2.drawContours(im_inv, contours, -1, (20,255,60), 1)  #draw contours
if plot == True:
    plots[9].axis('off')
    plots[9].imshow(im_inv, cmap=plt.cm.bone)
