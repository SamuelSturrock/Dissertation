import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree

DATADIR = 'M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\DataSets\\EmotionsFullFaceFer2013'
DATADIR2 = 'M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\DataSets\\EmotionsFullFace'
CATEGORIES = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

IMG_SIZE = 90
height =90

def create_training_data(data):
    for category in CATEGORIES:
        path = os.path.join(data, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                new_array = cv2.resize(img_array, (IMG_SIZE, height))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


training_data = []
# create_training_data(DATADIR)
create_training_data(DATADIR2)

lenofimage = len(training_data)

X = []
y = []

for categories, label in training_data:
    X.append(categories)
    y.append(label)

print(np.array(X).shape)
X = np.array(X).reshape(lenofimage, -1)

print(X.shape)

X = X / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)





fig = plt.figure(figsize=(100,100))

DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(X_train, y_train)
y2 = DecisionTree.predict(X_test)
print(y2)



print("Accuracy:", accuracy_score(y_test, y2))






