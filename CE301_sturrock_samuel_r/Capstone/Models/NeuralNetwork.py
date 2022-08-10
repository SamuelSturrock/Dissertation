import os

import joblib
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
file = open('results.csv', 'w', newline='')
file.writer = csv.writer(file)


DATADIR = ['M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\DataSets\\EmotionsFullFaceFer2013']

models = ["M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\Capstone\\Live\\FullFaceLR"]
sz = [48]
for x in range(2):
    CATEGORIES = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']


    IMG_SIZE = sz[x]
    height = sz[x]



    file.writer.writerow((models[x],len(DATADIR[x])))
    file.writer.writerow(("Overall",'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'))

    a = models[x]


    def create_training_data(data):
        training_data = []
        for category in CATEGORIES:
            path = os.path.join(data, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    image = cv2.imread(os.path.join(path, img))  # , cv2.COLOR_BGR2GRAY
                    new_array = cv2.resize(image, (IMG_SIZE, round(height)))
                    # new_array = tf.expand_dims(new_array, axis=0)
                    training_data.append([new_array, class_num])
                except Exception as e:
                    print(e)
                    pass
        return training_data



    training_data = create_training_data(DATADIR[x])


    X = []
    y = []

    for categories, label in training_data:
        X.append(categories)
        y.append(label)
    X = np.array(X)
    X = X/255.0

    y = np.array(y)

    X_train ,X_test, y_train, y_test = train_test_split(X, y)

    angerY = []
    angerX = []
    disgustY = []
    disgustX = []
    fearY = []
    fearX = []
    happinessY = []
    happinessX = []
    neutralY = []
    neutralX = []
    sadnessY = []
    sadnessX = []
    surpriseY = []
    surpriseX = []

    for x in range(0, len(y_test)):
        i = y_test[x]
        j = X_test[x]
        if y_test[x] == 0:
            angerY.append(i)
            angerX.append(j)
        if y_test[x] == 1:
            disgustY.append(i)
            disgustX.append(j)
        if y_test[x] == 2:
            fearY.append(i)
            fearX.append(j)
        if y_test[x] == 3:
            happinessY.append(i)
            happinessX.append(j)
        if y_test[x] == 4:
            neutralY.append(i)
            neutralX.append(j)
        if y_test[x] == 5:
            sadnessY.append(i)
            sadnessX.append(j)
        if y_test[x] == 6:
            surpriseY.append(i)
            surpriseX.append(j)

    angerX = np.array(angerX)
    disgustX = np.array(disgustX)
    fearX = np.array(fearX)
    happinessX = np.array(happinessX)
    neutralX = np.array(neutralX)
    sadnessX = np.array(sadnessX)
    surpriseX = np.array(surpriseX)

    angerY = np.array(angerY)
    disgustY = np.array(disgustY)
    fearY = np.array(fearY)
    happinessY = np.array(happinessY)
    neutralY = np.array(neutralY)
    sadnessY = np.array(sadnessY)
    surpriseY = np.array(surpriseY)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=500)

    test_loss, overall  = model.evaluate(X_test, y_test, verbose=2)
    test_loss, anger = model.evaluate(angerX, angerY, verbose=2)
    test_loss, disgust = model.evaluate(disgustX, disgustY  , verbose=2)
    test_loss, fear = model.evaluate(fearX, fearY, verbose=2)
    test_loss, happiness = model.evaluate(happinessX, happinessY, verbose=2)
    test_loss, neutral = model.evaluate(neutralX, neutralY, verbose=2)
    test_loss, sadness = model.evaluate(sadnessX, sadnessY, verbose=2)
    test_loss, surprise = model.evaluate(surpriseX, surpriseY, verbose=2)
    


    file.writer.writerow((a))
    file.writer.writerow((overall,anger, disgust, fear, happiness, neutral, sadness, surprise))




    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])




    model.save(a+".model", save_format="h5")
    print("Saved")
