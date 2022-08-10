import csv
import cv2
import random
import pathlib

# Load data
from matplotlib import pyplot as plt

savveDS = "M:\\Documents\\3\\CE301\\CE301_sturrock_samuel_r\\AudioVisualClipSavve"
data_dir = pathlib.Path(savveDS)

# Separate data based on actor
DC = list(data_dir.glob('DC/*.avi'))
JE = list(data_dir.glob('JE/*.avi'))
JK = list(data_dir.glob('JK/*.avi'))
KL = list(data_dir.glob('KL/*.avi'))

# Separate data based on emotion
anger = list(data_dir.glob('*/a*.avi'))
disgust = list(data_dir.glob('*/d*.avi'))
fear = list(data_dir.glob('*/f*.avi'))
happy = list(data_dir.glob('*/h*.avi'))
sad = list(data_dir.glob('*/sa*.avi'))
surprise = list(data_dir.glob('*/su*.avi'))
neutral = list(data_dir.glob('*/n*.avi'))

# Get samples of data
anger
disgust = disgust[:10]
fear = fear[:10]
happy = happy[:10]
sad = sad[:10]
surprise = surprise[:10]
neutral = neutral[:10]

# Create variable with all samples included
master = anger + disgust + fear + neutral + happy + sad + surprise

print(len(anger))

# Function to retrieve actor in vid
def getPerson(E):
    if E in DC:
        return "DC"
    if E in JE:
        return "JE"
    if E in JK:
        return "JK"
    if E in KL:
        return "KL"


# Removes recently played vid from scope and returns emotion and actor to CSV writer
def popE(E):
    if E in happy:
        happy.remove(E)
        master.remove(E)
        return "happy" + " " + getPerson(E)
    if E in sad:
        sad.remove(E)
        master.remove(E)
        return "Sad" + " " + getPerson(E)
    if E in neutral:
        neutral.remove(E)
        master.remove(E)
        return "Neutral" + " " + getPerson(E)
    if E in disgust:
        disgust.remove(E)
        master.remove(E)
        return "Disgust" + " " + getPerson(E)
    if E in fear:
        fear.remove(E)
        master.remove(E)
        return "Fear" + " " + getPerson(E)
    if E in surprise:
        surprise.remove(E)
        master.remove(E)
        return "Surprise" + " " + getPerson(E)
    if E in anger:
        anger.remove(E)
        master.remove(E)
        return "Anger" + " " + getPerson(E)


# Main Function. Plays video, collects human input and writes it to CSV
with open('../HumanEmotionRecDataOG.csv', 'w', newline='') as csvfile:
    file = csv.writer(csvfile)
    file.writerow(['Emotion', 'Human Prediction', "PinD"])

    x = 0
    while master:
        print("Enter the emotion you think it is: Happy, Sad, Neutral, Disgust, Fear, Anger, Surprise")

        rVid = random.choice(master)
        cap = cv2.VideoCapture(str(rVid))
        ret, frame = cap.read()

        while (1):
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
                cap.release()
                cv2.destroyAllWindows()
                break
            cv2.imshow('frame', frame)

        emotion = popE(rVid)
        emotion = emotion.split()

        cap.release
        v = input("What emotion do you think this is?")

        v.lower()
        file.writerow([emotion[0], v, emotion[1]])
        x += 1


def Grapher():
    content = []
    correct = 0
    wrong = 0

    fearC = 0
    fearW = 0

    with open('../HumanEmotionRecDataOG.csv', 'r', newline='') as csvfile:
        Rfile = csv.reader(csvfile)

        for row in Rfile:
            content.append(row)

        for i in range(len(content)):
            if content[i][0].lower() == content[i][1].lower():
                correct += 1
            else:
                wrong += 1

    with open('../HumanEmotionRecDataOG.csv', 'w', newline='') as csvfile:
        file = csv.writer(csvfile)

        for i in content:
            file.writerow(i)

    print(correct, wrong)

    labels = 'Correct', 'Wrong'
    sizes = [correct, wrong]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
