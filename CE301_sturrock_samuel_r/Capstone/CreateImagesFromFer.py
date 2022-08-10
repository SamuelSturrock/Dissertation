import os

import pandas
import cv2
import numpy as np

ferFilePath = "M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\PureDataSets\\fer203\\"


# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
def getEmotion(data):
    if data == 0:
        return "anger"
    if data == 1:
        return "disgust"
    if data == 2:
        return "fear"
    if data == 3:
        return "happiness"
    if data == 4:
        return "sadness"
    if data == 5:
        return "surprise"
    if data == 6:
        return "neutral"


images = pandas.read_csv("fer2013.csv")

for i in range(images.shape[0] - 1):


    image = images.iloc[i + 1, 1].split()
    image = list((map(int, image)))
    image = np.reshape(np.ravel(image), (48, 48))
    print(type(image))
    emotion = getEmotion(images.iloc[i + 1, 0])
    cv2.imwrite(os.path.join(ferFilePath, emotion + "\\" + emotion + str(i + 1) + ".png"), image)



