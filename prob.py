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
import glob
import csv
from scipy import ndimage as ndi
#img = cv2.imread('project/img/dil/test193.png',0)

img_dir = "/home/mca-pc74/proj/new/"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
#data = []
i=0
ar=[]
for f1 in files:
    if(f1.startswith("/home/mca-pc74/proj/new/1")):
        i=i+1
print("Number:",i)
for f1 in files:
    if(f1.startswith("/home/mca-pc74/proj/new/1")):
        im = cv2.imread(f1,0)
        img = cv2.getRectSubPix(im, (320, 240), (150, 150))

#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])

#plt.show()

        (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY |
        cv2.THRESH_OTSU)
        thresh = 136
        im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
 
        cleared = clear_border(im_bw)

        label_image = label(cleared)


        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0



    
        selem = disk(10)
        binary = binary_closing(binary, selem)

        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)

        get_high_vals = binary == 0
        img[get_high_vals] = 0

        image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        numOfContours = len(contours)   #number of contours

        area = []
   
        count = 0
        for count in range(numOfContours) :
            cv2.drawContours(img, contours, -1, (20,255,60), 1)  #draw contours
            cnt = contours[count]
            area.append(cv2.contourArea(cnt))
    #print(area)
    
    
            count+=1
    #print(contours)

#print(numOfContours)    
        if len(area)==0:
            a=0
        
        else:
        
            a=max(area)
        #print(x)
        ar.append(a)  
print("total:",len(ar))
w=sum(ar)
avg=w/i
d = np.sqrt(i)
div=avg/d
threshold=div+d
print("thresh:",threshold)
for f1 in files:
    if(f1.startswith("/home/mca-pc74/proj/new/0")):
        im = cv2.imread(f1,0)
        img = cv2.getRectSubPix(im, (320, 240), (150, 150))

#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])

#plt.show()

        (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY |
        cv2.THRESH_OTSU)
        thresh = 136
        im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
 
        cleared = clear_border(im_bw)

        label_image = label(cleared)


        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0



    
        selem = disk(10)
        binary = binary_closing(binary, selem)

        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)

        get_high_vals = binary == 0
        img[get_high_vals] = 0

        image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        numOfContours = len(contours)   #number of contours

        area = []
   
        count = 0
        for count in range(numOfContours) :
            cv2.drawContours(img, contours, -1, (20,255,60), 1)  #draw contours
            cnt = contours[count]
            area.append(cv2.contourArea(cnt))
    #print(area)
    
    
            count+=1
    #print(contours)

#print(numOfContours)    
        if len(area)==0:
            a=0
        
        else:
        
            a=max(area)
        if a>threshold:
          
            newname = f1.replace('/home/mca-pc74/proj/new/0', '/home/mca-pc74/proj/new/00p')
            os.rename(f1, newname)   
        
