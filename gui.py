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

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
import imutils
import operator
import random
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import Tk, Menu, Canvas
from PIL import Image, ImageTk
#import tkFileDialog as filedialog
from tkinter import filedialog
def reading(imageSource):
    global predictions_test
    im = cv2.imread(imageSource,0)
    img = cv2.getRectSubPix(im, (320, 240), (150, 150))
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



    selem = disk(2)
    binary = binary_erosion(binary, selem)    
    selem = disk(10)
    binary = binary_closing(binary, selem)

    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    get_high_vals = binary == 0
    img[get_high_vals] = 0

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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
    contours, hierarchy = cv2.findContours(im_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    #print(numOfContour) 
    if ar == []:
        area=0
    else:
        area=sum(ar)

    csvTitle = [['NoduleArea' ,'NoduleContour','Perimeter','Eccentricity','Diameter','LungArea','LungContour']]
    csvData = []
    csvData.append([a,numOfContours, p, e,d,area,numOfContour])
    with open('new.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvTitle)
        writer.writerows(csvData)


    d=pd.read_csv("new.csv")
    d_new=pd.read_csv("new.csv",na_values=['?'])

    d_new.dropna(inplace=True)

    X_test=d_new[['NoduleArea','Perimeter','Diameter' ,'Eccentricity']]

    predictions_test = svc.predict(X_test)
    if predictions_test==0:
        print("Cancer not detected")
    if predictions_test==1:
        print("Cancer detecetd \n Stage:1")
    if predictions_test==2:
        print("Cancer detected \n Stage:2")
    if predictions_test==3:
        print("Cancer detected \n Stage:3")
    #print("prediction:",predictions_test)

def do_processing(path_to_file):
    global labell
    # load the image
    image = cv2.imread(path_to_file)
    orig = image.copy()
    image = cv2.getRectSubPix(image, (320, 240), (150, 150))
    thresh = 136
    image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    
    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("model_lungcanc")
     
    # classify the input image
    (no,first,second,third) = model.predict(image)[0]

    # build the label
    label_dict = {
        "no cancer detected": no,
        "cancer first stage": first,
	    "cancer second stage": second,
	    "cancer third stage": third
    }

    labell = max(label_dict.items(), key=operator.itemgetter(1))[0]
    max_value = label_dict[labell]
    x=random.randint(1,20)

    labell = "{}".format(labell)
     
    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, labell, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	    0.7, (0, 255, 0), 2)
     
    # show the output image
    #cv2.imwrite("op.jpg",output)
    
    plt.subplot(121),plt.imshow(orig, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    print("Result:",labell)
width = 430
height = 320
# Function definitions
def deleteImage(canvas):
    canvas.delete("all")
    return


def quitProgram():
    gui.destroy()


# Main window
gui = Tk()
gui.title("Prediction")
gui.geometry("800x500+320+100")
gui.configure(background="white")
width=400
height=200
screen_width=gui.winfo_screenwidth()
screen_height=gui.winfo_screenheight()
x_cord=(screen_width/2)-(width/2)
y_cord=(screen_height/2)-(height/2)

# setting the size of the window



# BROWSE BUTTON CODE
def browsefunc():
    global imgname
    imgname = filedialog.askopenfilename()
    pathLabel = Label(gui, text=imgname, anchor=N )
    pathLabel.pack()
    width = 430
    height = 320
    img = Image.open(imgname)
    im2 = img.resize((250, 250), Image.NEAREST)
    #width, height = img.size
    filename = ImageTk.PhotoImage(im2)
    canvas.image = filename  # <--- keep reference of your image
    canvas.create_image(600,0,anchor=N,image=filename)
    canvas.pack()


# Inside the main gui window
# Creating an object containing an image
# A canvas with borders that adapt to the image within it
def loadImage():

    width = 430
    height = 320
    img = Image.open(imgname)
    im2 = img.resize((250, 250), Image.NEAREST)
    #width, height = img.size
    filename = ImageTk.PhotoImage(im2)
    canvas.image = filename  # <--- keep reference of your image
    canvas.create_image(600,0,anchor=N,image=filename)
    canvas.pack()
#RUNNING PROGRAM
def run():
    reading(imgname)
    popup = Tk()
    popup.wm_title("Result")
    popup.geometry("220x100+420+300")
    popup.configure(background="white")
    width=200
    height=100
    screen_width=popup.winfo_screenwidth()
    screen_height=popup.winfo_screenheight()
    x_cord=(screen_width/2)-(width/2)
    y_cord=(screen_height/2)-(height/2)
    pred = predictions_test
    if(pred==0):
        predLabel = Label(popup, text = "Not Cancerous..", anchor = S)
    elif(pred== 1):
        predLabel = Label(popup, text = " Cancer stage 1\n\n Get treatment..", anchor = S)
    elif(pred== 2):
        predLabel = Label(popup, text = " Cancer stage 2\n\n Meet the doctor immediately..", anchor = S)
    elif(pred==3):
        predLabel = Label(popup, text = " Cancer stage 3 \n\n Meet the doctor immediately..", anchor = S)
    predLabel.configure(background="white")
    predLabel.pack(side="top", fill="x", pady=10)
    B1 = Button(popup, text="CLOSE", command = popup.destroy)
    B1.configure(background="white")
    B1.pack()
    popup.mainloop()
    
def runcode():
    do_processing(imgname)
    pop = Tk()
    pop.wm_title("Result")
    pop.geometry("220x100+420+300")
    pop.configure(background="white")
    width=200
    height=100
    screen_width=pop.winfo_screenwidth()
    screen_height=pop.winfo_screenheight()
    x_cord=(screen_width/2)-(width/2)
    y_cord=(screen_height/2)-(height/2)
    #print(labell)
    #pr = Label(pop, text = labell, anchor = S)
    if(labell=="no cancer detected"):
        pr = Label(pop, text = "Not Cancerous..", anchor = S)
    elif(labell=="cancer first stage"):
        pr = Label(pop, text = " Cancer detected (stage 1)\n\n Get treatment..", anchor = S)
    elif(labell=="cancer second stage"):
        pr = Label(pop, text = " Cancer detected (stage 2) \n\n Meet the doctor immediately..", anchor = S)
    elif(labell=="cancer third stage"):
        pr = Label(pop, text = " Cancer detected(stage 3)\n\n Meet the doctor immediately..", anchor = S)
    pr.configure(background="white")
    pr.pack(side="top", fill="x", pady=10)
    B = Button(pop, text="CLOSE", command = pop.destroy)
    B.configure(background="white")
    B.pack()
    pop.mainloop()
    

la = Label(gui, text = "LUNG CANCER DETECTION SYSTEM",pady=10, anchor = S)
la.config(font=("Courier", 15))
la.configure(background="white")
la.pack()
label2 = Label(gui, text = "",pady=10, anchor = S)
label2.configure(background="white")
label2.pack()

browsebutton = Button(gui, text="Browse",padx=47, command=browsefunc)
browsebutton.configure(background="white")
browsebutton.pack()
pathlab= Label(gui)
pathlab.configure(background="white")
pathlab.pack()
#loadbut =Button(gui,text="LOAD THE IMAGE",padx=18,fg="red",command=loadImage)
#loadbut.configure(background="white")
#loadbut.pack()
#print("filenma",file)
button = Button(gui,text="RESULT USING SVM",fg="red",command=run)
button.configure(background="white")
button.pack()
label6 = Label(gui, text = "", pady=2,anchor = S)
label6.configure(background="white")
label6.pack()

#print("filenma",file)
but =Button(gui,text="RESULT USING CNN",fg="red",command=runcode)
but.configure(background="white")
but.pack()
label5 = Label(gui, text = "", pady=2,anchor = S)
label5.configure(background="white")
label5.pack()

qbut =Button(gui,text="QUIT",fg="red",padx=53,command=quitProgram)
qbut.configure(background="white")
qbut.pack()
label4 = Label(gui, text = "", pady=2,anchor = S)
label4.configure(background="white")
label4.pack()
canvas = Canvas(gui, width = 800, height =300)
canvas.configure(background="white")
canvas.pack()
gui.mainloop()
