import glob
import pathlib
import cv2
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

CKPLUSDS = pathlib.Path("M:\\Documents\\3\\CE301\CE301_sturrock_samuel_r\\PureDataSets\\CK+\\")
fer2013 = pathlib.Path("M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\PureDataSets\\fer203\\")
emotions = ["anger","disgust","fear","happiness","neutral","sadness","surprise"]

fullFace = "M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\DataSets\\EmotionsFullFaceFer2013\\"
croppedFace = "M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\DataSets\\EmotionsCropFer2013\\"

def ProcessDataSet(EmotionSet, emotion):
    i = 0
    for img in EmotionSet:
        image = cv2.imread(str(img),cv2.IMREAD_GRAYSCALE)
        face = cropWholeFace(image)
        try:
            h,w = face.shape
            upperFace = face[0:int(h/2), 0:w]
            SaveFiles(face,upperFace,emotion, i)
            i +=1
        except:
            continue





def cropWholeFace(image):
    face = face_cascade.detectMultiScale(image, 1.1, 4)

    for (x, y, w, h) in face:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (48,48), interpolation=cv2.INTER_LINEAR)
        return face


def SaveFiles(wFace,cFace,emotion,i):
    cv2.imwrite(os.path.join(fullFace,emotion + "\\" + emotion+str(i)+".png"), wFace)
    cv2.imwrite(os.path.join(croppedFace,emotion + "\\" + emotion+str(i)+".png"), cFace)



# for x in emotions:
#     print(len(list(CKPLUSDS.glob(x+ "\\*"))),x)

for x in emotions:
    ProcessDataSet(list(fer2013.glob(x+ "\\*")),x)