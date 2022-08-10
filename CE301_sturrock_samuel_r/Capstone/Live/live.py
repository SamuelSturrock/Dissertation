import math
import pickle

import joblib
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
glob = 0




model = keras.models.load_model("M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\Capstone\\Live\\FullFaceLR.model")


font                   = cv2.FONT_HERSHEY_SIMPLEX

fontScale              = .5
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2
CATEGORIES = ['anger', 'disgust',      'fear', 'happiness', 'neutral', 'sadness', 'surprise']
ColoursDict = {(0,0,255): "Anger",
               (0, 255, 0): "Disgust",
               (0, 0, 0): "Fear",
               (0, 255, 255): "Happiness",
               (255, 255, 255): "Neutral",
               (255, 0, 0): "Sadness",
               (100, 25, 255): "Surprise"}
def LRCROPred(image):
    image = cv2.resize(image, (48,48))
    image = image[0:48,0:48]
    image = image/255
    image = tf.expand_dims(image,0)
    image = model.predict(image)
    score = tf.nn.softmax(image)
    cat =  CATEGORIES[np.argmax(score)]
    return(cat)

def EmColour(image):
    if image == "anger":
        return (0,0,255)
    if image == "disgust":
        return (0, 255, 0)
    if image == "fear":
        return (0, 0, 0)
    if image == "happiness":
        return (0, 255, 255)
    if image == "neutral":
        return (255, 255, 255)
    if image == "sadness":
        return (255, 0, 0)
    if image == "surprise":
        return (100, 25, 255)


def getPercentages(list, emos,width):
    try:
        print(emos/list*100)

        return (emos/list) *100
    except:
        return 0







# define a video capture object


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
vid = cv2.VideoCapture(0)

while (True):


    ret, frame = vid.read()


    # Display the resulting frame

    height, width, _ = frame.shape


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4,0,(15,15))
    # Draw the rectangle around each face
    EmoList = []
    for (x, y, w, h) in faces:

        face = frame[y:y + h, x:x + w]
        emo = LRCROPred(face)
        EmoList.append(emo)
        colour = EmColour(emo)
        cv2.putText(frame, str(emo), (x,y), font, fontScale, colour,thickness, lineType)
        cv2.rectangle(frame, (x, y+5), (x + w, y + h), colour, 2)


    val = 0
    for x in CATEGORIES:
        emotion = EmoList.count(x)
        val+= 40
        if emotion == 0:
            percent = 0
        else:
            percent = getPercentages(len(EmoList),emotion,width)
        colour = EmColour(x)
        cv2.putText(frame, str(ColoursDict[colour]), (width-100, 0+val-7), font, fontScale, colour, thickness, lineType)
        cv2.line(frame, (width-100, 0+val), (width, 0+val), (0,0,0), 10)

        cv2.line(frame, (width-100, 0+val), (round(width-100 + percent), 0+val), colour, 10)




        if cv2.waitKey(1) & 0xFF == ord("c"):
            glob +=1
            if glob ==4:
                glob = 0


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('frame', frame)



# After the loop release the cap object
vid.release()

cv2.destroyAllWindows()