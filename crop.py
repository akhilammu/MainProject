import cv2
import os
import glob
img_dir = "C:/users/hp/image/" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

for f1 in files:
    img = cv2.imread(f1)
    img = cv2.getRectSubPix(img, (320, 220), (150, 170))
    cv2.imwrite(f1, img)
