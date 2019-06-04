import numpy as np
import csv
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



with open('./Model/A_Z_HR_Data.csv','r') as csv_file:
    result = csv.reader(csv_file)
    rows = []

    for row in result:
        rows.append(row)

#Training Data
train_data = []
train_label =[]
for letter in rows:
    if (letter[0] == '0') or (letter[0] == '1') or (letter[0] == '2') or (letter[0] == '3'):
        x = np.array([int(j) for j in letter[1:]])
        x = x.reshape(28, 28)
        train_data.append(x)
        train_label.append(int(letter[0]))
    else:
        break
#print(len(train_label))
print(len(train_data))
print(len(train_label))

letter = rows[60000]
x = np.array([int(j) for j in letter[1:]])
x = x.reshape(28, 28)
#
shuffle_order = list(range(56081))
random.shuffle(shuffle_order)

train_data = np.array(train_data)
train_label = np.array(train_label)

train_data = train_data[shuffle_order]
train_label = train_label[shuffle_order]

#Generate Data

print(train_data.shape)
train_x = train_data[:50000]
train_y = train_label[:50000]

val_x = train_data[50000:53000]
val_y = train_label[50000:53000]

test_x = train_data[53000:]
test_y = train_label[53000:]
#Generate Const

BATCH_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 128
LR = 0.001
N_EPOCHS = 1

#Tflearn

tf.reset_default_graph()

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1])

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2) #3

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

#Process data
#1 Train data

train_x = train_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_x = val_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x = test_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#2 Test data

original_test_y = test_y

train_y = to_categorical(train_y, N_CLASSES)
val_y = to_categorical(val_y, N_CLASSES)
test_y = to_categorical(test_y, N_CLASSES)

#Training

model.load('./handwriting_128.tflearn')

model.fit(train_x, train_y, n_epoch=N_EPOCHS, validation_set=(val_x, val_y), show_metric=True)
model.save('handwriting_128.tflearn')




# dự đoán với tập dữ liệu test
test_logits = model.predict(test_x)
#lấy phần tử có giá trị lớn nhất
test_logits = np.argmax(test_logits, axis=-1)
print(np.sum(test_logits == original_test_y) / len(test_logits))