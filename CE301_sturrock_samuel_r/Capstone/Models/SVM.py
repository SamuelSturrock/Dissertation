import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import metrics

DATADIR = 'M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\DataSets\\EmotionsFullFaceFer2013'
DATADIR2 = 'M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\DataSets\\EmotionsFullFace'
CATEGORIES = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

IMG_SIZE = 500
height =500

def create_training_data(data):
    for category in CATEGORIES:
        path = os.path.join(data, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)#
                new_array = cv2.resize(img_array, (IMG_SIZE, height))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


training_data = []
create_training_data(DATADIR2)

lenofimage = len(training_data)

X = []
y = []

for categories, label in training_data:
    X.append(categories)
    y.append(label)
X = np.array(X).reshape(lenofimage, -1)


X = X / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

angerY = []
angerX = []
disgustY= []
disgustX = []
fearY= []
fearX = []
happinessY= []
happinessX = []
neutralY= []
neutralX = []
sadnessY= []
sadnessX = []
surpriseY= []
surpriseX = []


for x in range(0,len(y_test)):
    i = y_test[x]
    j = X_test[x]
    if y_test[x] ==0:
        angerY.append(i)
        angerX.append(j)
    if y_test[x] ==1:
        disgustY.append(i)
        disgustX.append(j)
    if y_test[x] ==2:
        fearY.append(i)
        fearX.append(j)
    if y_test[x] ==3:
        happinessY.append(i)
        happinessX.append(j)
    if y_test[x] ==4:
        neutralY.append(i)
        neutralX.append(j)
    if y_test[x] ==5:
        sadnessY.append(i)
        sadnessX.append(j)
    if y_test[x] ==6:
        surpriseY.append(i)
        surpriseX.append(j)


svc = SVC(kernel='rbf', gamma='auto')
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)

print(accuracy_score(y_test,predicted))


# test_loss, test_acc = accuracy_score(angerX,  angerY)
# print("anger=",test_acc)
# test_loss, test_acc = model.evaluate(disgustX,  disgustY, verbose=2)
# print("disgust",test_acc)
# test_loss, test_acc = model.evaluate(fearX,  fearY, verbose=2)
# print("fear",test_acc)
# test_loss, test_acc = model.evaluate(happinessX,  happinessY, verbose=2)
# print("happiness",test_acc)
# test_loss, test_acc = model.evaluate(neutralX,  neutralY, verbose=2)
# print("neutral",test_acc)
# test_loss, test_acc = model.evaluate(sadnessX,  sadnessY, verbose=2)
# print("sadness",test_acc)
# test_loss, test_acc = model.evaluate(surpriseX,  surpriseY, verbose=2)
# print("surprise",test_acc)


