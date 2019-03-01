import cv2
import os
import glob
img_dir = "C:/users/hp/image/" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

for f1 in files:
    im_gray = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)
    #data.append(img)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(f1, im_bw)
