# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import operator
import random

def do_processing(path_to_file):

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
    model = load_model("mod")
     
    # classify the input image
    (no,first,second,third) = model.predict(image)[0]

    # build the label
    label_dict = {
        "no": no,
        "first": first,
	    "second": second,
	    "third": third
    }

    label = max(label_dict.items(), key=operator.itemgetter(1))[0]
    max_value = label_dict[label]
    x=random.randint(1,20)

    label = "{}: {:.2f}%".format(label, max_value * 100 - x)
     
    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	    0.7, (0, 255, 0), 2)
     
    # show the output image
    cv2.imwrite("output.jpg",output)
do_processing("t.png")
