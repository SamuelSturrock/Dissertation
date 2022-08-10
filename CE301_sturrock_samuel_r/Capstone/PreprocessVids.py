import glob
import pathlib
import random
import PIL
import cv2
import os
from PIL import Image
import numpy
import numpy as np
import imageio

# Import face detector & DS
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
savveDS = "M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\AudioVisualClipSavve"

data_dir = pathlib.Path("M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\Emotions")

emotionFiles = list(data_dir.glob("*/*"))

# Separate data based on emotion
data_dir = pathlib.Path(savveDS)

DC = list(data_dir.glob('DC/*.avi'))
JE = list(data_dir.glob('JE/*.avi'))
JK = list(data_dir.glob('JK/*.avi'))
KL = list(data_dir.glob('KL/*.avi'))

trainingPeople = DC + JE
TestingPeople = JK + KL

Anger = list(data_dir.glob('*/a*.avi'))
Disgust = list(data_dir.glob('*/d*.avi'))
Fear = list(data_dir.glob('*/f*.avi'))
Happy = list(data_dir.glob('*/h*.avi'))
Sad = list(data_dir.glob('*/sa*.avi'))
Surprise = list(data_dir.glob('*/su*.avi'))
Neutral = list(data_dir.glob('*/n*.avi'))

# master = anger + disgust + fear + neutral + happy + sad + surprise

emVids = []


# Saves image data to a file and resizes it so they all have the same dimensions
def saveFiles(E, A):
    os.chdir("M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\TrainingPlp\\" + A)
    i = 0
    print(len(E))
    while True:
        try:
            for i in range(len(E)):
                l = 0
                try:
                    for l in range(len(E[i])):
                        try:
                            a = str(i) + "f" + str(l) + ".png"
                            imageio.imwrite(a, E[i][l])
                            l += 1
                            image = PIL.Image.open(a)
                            new_image = image.resize((165, 165))
                            new_image.save("M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\TrainingPlp\\" + A + "\\" +a+".png", 'png')
                        except ValueError:
                            print(ValueError)
                            pass

                except ValueError:
                    print(ValueError)
                    pass

                print(i)
        except ValueError:
            print(ValueError)
            pass

        break


# Pre-processes videos in list.


def getEmotionDS(emotion):
    z = 0
    for x in range(len(emotion)):
        cap = cv2.VideoCapture(str(emotion[x]))

        if not (cap.isOpened()):
            print("Error reading file")

        # Check if you are able to capture the video
        ret, fFrame = cap.read()
        vid = []

        # split the video into frames. Greyscale & crops image to only contain the faces
        while ret == True:
            ret, fFrame1 = cap.read()
            try:
                img1 = fFrame1.copy()
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                a = face_cascade.detectMultiScale(img1)
                for (i, y, w, h) in a:
                    cv2.rectangle(img1, (i, y), (i + w, y + h), (0, 0, 255), 2)
                    img1 = img1[y:y + h, i:i + w]
                    vid.append(img1)

            except:

                break;
        z = z + 1

        emVids.append(vid)
        print("Has been Appended " + str(z))

    return emVids


# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# print(len(vid))

def crop_images(files):
    for x in files:
        x = str(x)
        try:
            im = Image.open(x)
            im1 = im.crop((0, 0, 165, 83))
            im1.save(x)
        except:
            pass


# anger = pre.getEmotionDS(pre.anger)
# sad  = pre.getEmotionDS(pre.sad)
# neutral  = pre.getEmotionDS(pre.neutral)
# happy  = pre.getEmotionDS(pre.happy)
# surprise  = pre.getEmotionDS(pre.surprise)
# disgust  = pre.getEmotionDS(pre.disgust)
# fear  = pre.getEmotionDS(pre.fear)

# master = #anger  + sad + neutral + happy + surprise + disgust + fear

from PIL import Image

from PIL import Image
from matplotlib import cm


# print(len(sad))

# pre.saveFiles(neutral, "Neutral")
# pre.saveFiles(happy, "Happy")
# pre.saveFiles(surprise, "Surprise")
# pre.saveFiles(disgust, "Disgust")
# pre.saveFiles(fear, "Fear")
# pre.saveFiles(anger, "Anger")
# pre.saveFiles(sad, "Sad")
# pre.crop_images(pre.emotionFiles)


def crop_videos():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    CroppedImages = "M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\EmotionsCrop\\"
    data_dir = str(pathlib.Path(CroppedImages))

    Emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    size = 0

    img_array = []
    for emotion in Emotions:
        x = 0

        for filename in glob.glob(data_dir + emotion + "\\" + str(x) + "*"):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        x += 1
        out = cv2.VideoWriter(emotion + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for j in range(len(img_array)):
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


emotion = [Anger, Disgust, Fear, Neutral, Happy, Sad, Surprise]
print(eval("emotion"))

for i in emotion:
    trainPLP = []
    TestPLP = []
    for x in i:
        if x in trainingPeople:
            trainPLP.append(x)

        if x in TestingPeople:
            trainPLP.append(x)

    trainPLP = getEmotionDS(trainPLP)

    i = [j for j, a in locals().items() if a == i][0]

    saveFiles(trainPLP, i)
