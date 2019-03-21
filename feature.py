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
fileNameList = []
image_list = os.listdir("/home/mca-pc74/proj/new/")
for files in image_list:
    fileName, extension = os.path.splitext(files)
    fileNameList.append(fileName)
img_dir = "/home/mca-pc74/proj/new/"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
#data = []
csvTitle = [['No', 'NoduleArea','NoduleContour' ,'Perimeter','Eccentricity','Diameter','LungArea','LungContour','prediction']]
csvData = []
x=1
for f1 in files:
    if(f1.startswith("/home/mca-pc74/proj/new/00p")):
        pred=10
    elif f1.startswith("/home/mca-pc74/proj/new/0"): 
        pred=0
    elif f1.startswith("/home/mca-pc74/proj/new/1"):
        pred=1
    elif f1.startswith("/home/mca-pc74/proj/new/2"):
        pred=2
    elif f1.startswith("/home/mca-pc74/proj/new/3"):
        pred=3
    im = cv2.imread(f1,0)
    img = cv2.getRectSubPix(im, (320, 240), (150, 150))

#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])

#plt.show()

    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY |
    cv2.THRESH_OTSU)
    thresh = 136
    im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    ret,im_inv = cv2.threshold(im_bw,thresh,255,cv2.THRESH_BINARY_INV)
    
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
    perimeter=[]
    count = 0
    for count in range(numOfContours) :
        cv2.drawContours(img, contours, -1, (20,255,60), 1)  #draw contours
        cnt = contours[count]
        area.append(cv2.contourArea(cnt))
        peri = cv2.arcLength(cnt,True)
        perimeter.append(peri)
    #print(area)
    
    
        count+=1
    #print(contours)

#print(numOfContours)    
    if len(area)==0:
        a=0
        p=0
        e=0
        d=0
    else:
        
        a=max(area)
        #print(x)
        print("area:",a)   #gives the largest area
        for i in range(numOfContours) :
            if area[i]==a:
                k=i
        if a<20:
            e=1
        else:
            cnt = contours[k]
            ellipse = cv2.fitEllipse(cnt)
            (center,axes,orientation) = ellipse
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            e=minoraxis_length/majoraxis_length
        p=perimeter[k]
        d = np.sqrt(4*a/np.pi)
    im_inv = clear_border(im_inv)
    image, contours, hierarchy = cv2.findContours(im_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ar = []
    count = 0
    numOfContour = len(contours)
    for count in range(numOfContour) :
        cv2.drawContours(im_inv, contours, -1, (20,255,60), 1)  #draw contours
        cnt = contours[count]
        ar.append(cv2.contourArea(cnt))
    #print(area)
    
    
        count+=1
    #print(contours)

    print(numOfContour) 
    if ar == []:
        print(0)
        area=0
    else:
    #print(max(area))   #gives the largest area
        #print(area)
        print(sum(ar))
        area=sum(ar)
    print("perimeter:",p)
#print(eccentricity)
    print("eccentricity:",e)
    print("\n")
        #if a< 50:
        #    pred=0
        #elif e==1:
        #    pred=0
        #else:
        #    pred=1
    
    csvData.append([x, a,numOfContours, p, e,d,area,numOfContour,pred])
    x=x+1
    with open('features.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvTitle)
        writer.writerows(csvData)
c = cv2.waitKey(0);
if c == 27:           #wait for ESC key to exit
    cv2.destroyAllWindows();
