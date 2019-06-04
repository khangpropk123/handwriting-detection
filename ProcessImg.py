import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import cv2
import random

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

#Generate Const
BATCH_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 128
LR = 0.001
N_EPOCHS = 50

#Generate DNN
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1])

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, N_CLASSES, activation='softmax')
network = regression(network)

model = tflearn.DNN(network)

#Load Mode4
model.load("./handwriting_128.tflearn")
res = [] #Save

#Processing Image
img = cv2.imread('./Img/Test2.png', 0)
blur = cv2.GaussianBlur(img, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) #Convert Image To Binary
horizal = thresh
vertical = thresh
plt.imshow(imgqr)
plt.show()
#
scale_height = 30  #
scale_long = 15

long = int(img.shape[1] / scale_long)
height = int(img.shape[0] / scale_height)

horizalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (long, 1))
horizal = cv2.erode(horizal, horizalStructure, (-1, -1))
horizal = cv2.dilate(horizal, horizalStructure, (-1, -1))

verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

mask = vertical + horizal

#Get Answer Table
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max = -1
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > max:
        x_max, y_max, w_max, h_max = x, y, w, h
        max = cv2.contourArea(cnt)

table = img[y_max:y_max + h_max, x_max:x_max + w_max] # Answer Table
table1 = table[0:0+h_max,0:0+int(w_max/2)]
table2 = table[0:0+h_max,int(w_max/2):0+w_max]
thresh01 = cv2.adaptiveThreshold(table1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
thresh02 = cv2.adaptiveThreshold(table2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
y_max=0;
x_max=0;


#Process to get Input
cropped_thresh_img = []
cropped_origin_img = []
countours_img = []

NUM_ROWS = 26 # Number of row
START_ROW = 1
#
for i in range(START_ROW, NUM_ROWS):
    thresh1 = thresh01[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              x_max + round(w_max / 12):x_max + round(w_max / 4)]
    contours_thresh1, hierarchy_thresh1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    origin1 = table1[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              x_max + round(w_max / 12):x_max + round(w_max / 4)]

    cropped_thresh_img.append(thresh1)
    cropped_origin_img.append(origin1)
    countours_img.append(contours_thresh1)
   
   
for i in range(START_ROW, NUM_ROWS):
    thresh1 = thresh01[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              x_max + round(2 * w_max/6 ):x_max + round(w_max)]
    contours_thresh1, hierarchy_thresh1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    origin1 = table1[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              x_max + round(2 * w_max/6 ):x_max + round(w_max)]

    cropped_thresh_img.append(thresh1)
    cropped_origin_img.append(origin1)
    countours_img.append(contours_thresh1)
    
  

for i in range(START_ROW, NUM_ROWS):
    thresh1 = thresh02[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              x_max + round(w_max / 12):x_max + round(w_max / 4)]
    contours_thresh1, hierarchy_thresh1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    origin1 = table2[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              x_max + round(w_max / 12):x_max + round(w_max / 4)]

    cropped_thresh_img.append(thresh1)
    cropped_origin_img.append(origin1)
    countours_img.append(contours_thresh1)
    


for i in range(START_ROW, NUM_ROWS):
    thresh1 = thresh02[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              x_max + round(2 * w_max/6 ):x_max + round(w_max)]
    contours_thresh1, hierarchy_thresh1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    origin1 = table2[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              x_max + round(2 * w_max/6 ):x_max + round(w_max)]

    cropped_thresh_img.append(thresh1)
    cropped_origin_img.append(origin1)
    countours_img.append(contours_thresh1)
# Processing
num_Element = []
num_Space   = []
for i, countour_img in enumerate(countours_img):
    num_Element.append(i)
    for cnt in countour_img:
        if cv2.contourArea(cnt) > 30:
            x, y, w, h = cv2.boundingRect(cnt)
            if x > cropped_origin_img[i].shape[1] * 0.1 and x < cropped_origin_img[i].shape[1] * 0.9:
                answer = cropped_origin_img[i][y:y + h, x:x + w]
                answer = cv2.threshold(answer, 160, 255, cv2.THRESH_BINARY_INV)[1]
                answer = cv2.resize(answer,(28, 28))
                answer  = np.expand_dims(answer, 0)
                answer = np.expand_dims(answer, -1)
                res.append(np.argmax(model.predict(answer), axis=-1)) # Predict Answer
                num_Space.append(i)

#Get index of null
element_Lengh = len(num_Element)
for i in num_Space:
   num_Element.remove(i)
#print(num_Element)
letter = ['A', 'B', 'C', 'D']
#print(res)
result = []
for r in res:
    if len(r) == 0:
        result.append("X")
    elif len(r) > 1:
        result.append("O")
#"None"
    else:
        result.append(letter[int(r[0])])

for i in num_Element:
    result.insert(i,"None")

# Result: => result
print(result)

