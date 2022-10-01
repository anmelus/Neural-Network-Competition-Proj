import matplotlib.pyplot as plt
import random
import numpy as np
import os
# Subjects and Sessions
SUB1 = list(range(1,9))
SUB2 = list(range(1,6))
SUB3 = list(range(1,4))
SUB4 = list(range(1,3))
SUB5 = list(range(1,4))
SUB6 = list(range(1,4))
SUB7 = list(range(1,5))
SUB8 = list(range(1,2))
SUBS = [SUB1, SUB2, SUB3, SUB4, SUB5, SUB6, SUB7, SUB8]

X_HZ = 40
Y_HZ = 10

def nList(n):
    myList = []
    for i in range(n):
        myList.append([])
    return myList

def readCSV(path, type):
    with open(path, "r") as f:
        raw = f.readlines()
        numRows = len(raw[0].split(","))
        data = nList(numRows)
        for line in raw:  # Line by Line
            sLine = line.split(",")  # Split by commas
            for i in range(len(data)): # For each column
                data[i].append(type(sLine[i])) # Cast to type and append
    return data

def readData(folder, subject, session):
    dataPath = folder + "subject_00" + str(subject) + "_0" + str(session)
    data = {}  # Dictionary to hold all data
    # Read X Time
    tmp = readCSV(dataPath + "__x_time.csv", float)
    data["xtime"] = tmp[0]
    # Read X Data
    tmp = readCSV(dataPath + "__x.csv", float)
    data["ax"] = tmp[0]
    data["ay"] = tmp[1]
    data["az"] = tmp[2]
    data["gx"] = tmp[3]
    data["gy"] = tmp[4]
    data["gz"] = tmp[5]
    # Read Y Time
    tmp = readCSV(dataPath + "__y_time.csv", float)
    data["ytime"] = tmp[0]
    # Read Y Data
    tmp = readCSV(dataPath + "__y.csv", int)
    data["y"] = tmp[0]
    return data

def normalize(data):
    mean = np.mean(data,axis=0)
    sd = np.std(data, axis=0)
    data2 = (data - mean) / sd
    return data2

def normalize2(data):
    myrange = (abs(np.max(data)) + abs(np.min(data)))/2
    data2 = data / myrange
    return data2

def normailizeData(data):
    fields = ["ax", "ay", "az", "gx", "gy", "gz"]
    for field in fields:
        data[field] = normalize(data[field])
    return data


def plotData(data, subject, session):
    plt.figure()
    plt.suptitle("Subject: 00" + str(subject) + "   Session: 0" + str(session))
    plt.subplot(7,1,1)
    plt.title("X Accelerometer")
    plt.plot(data["xtime"], data["ax"])
    plt.subplot(7, 1, 2)
    plt.title("Y Accelerometer")
    plt.plot(data["xtime"], data["ay"])
    plt.subplot(7, 1, 3)
    plt.title("Z Accelerometer")
    plt.plot(data["xtime"], data["az"])

    plt.subplot(7, 1, 4)
    plt.title("X Gyroscope")
    plt.plot(data["xtime"], data["gx"])
    plt.subplot(7, 1, 5)
    plt.title("Y Gyroscope")
    plt.plot(data["xtime"], data["gy"])
    plt.subplot(7, 1, 6)
    plt.title("Z Gyroscope")
    plt.plot(data["xtime"], data["gz"])

    plt.subplot(7, 1, 7)
    plt.title("Labels")
    plt.plot(data["ytime"], data["y"])


def findSegmentEnd(data, numSamples, firstSegment, segmentTime):

    lastSegmentX = numSamples - firstSegment # last valid segment

    lastYTime = data["ytime"][-1]
    lastXTime = lastYTime + segmentTime
    i = len(data["xtime"]) - 1
    while data["xtime"][i] > lastSegmentX:
        i -= 1
    print(lastYTime, data["xtime"][i])
    return i

    #lastSegment = min(lastSegmentX, lastSegmentY)

# Finds the last X and Y times that align so that there is enough data to
# center y labels within a time segment
def findSegmentEnd2(data, segmentTime):
    lastYTime = data["ytime"][-1]
    lastXTime = data["xtime"][-1]
    lastXTimeI = len(data["xtime"]) - 1
    lastYTimeI = len(data["ytime"]) - 1
    # More Y than available X, find Y cutoff
    if lastYTime + segmentTime / 2 > lastXTime:
        while lastYTime + segmentTime / 2 > lastXTime:
            lastYTimeI -= 1
            lastYTime = data["ytime"][lastYTimeI]
    else: # More X Data than needed, find X cutoff
        while lastXTime > lastYTime + segmentTime / 2 :
            lastXTimeI += 1
            lastXTime = data["xtime"][lastXTimeI]
    print(lastYTime, data["xtime"][lastXTimeI])
    return lastXTimeI, lastYTimeI

# Finds the first X and Y times that align so that there is enough data to
# center y labels within a time segment
def findSegmentStart(data, segmentTime):
    firstYTime = data["ytime"][0]
    firstXTime = data["xtime"][0]
    firstXTimeI = 0
    firstYTimeI = 0
    # First X Time is 0, find first Y time
    if firstYTime - segmentTime / 2 < firstXTime:
        while firstYTime - segmentTime / 2 < firstXTime:
            firstYTimeI += 1
            firstYTime = data["ytime"][firstYTimeI]
    else:# First Y Time is 0, find first X time
        while firstXTime < firstYTime - segmentTime / 2 :
            firstXTimeI += 1
            firstXTime = data["xtime"][firstXTimeI]
    print(data["xtime"][firstXTimeI],firstYTime)
    return firstXTimeI, firstYTimeI

def getSegments(data, segmentTime, step, skipY=False):
    if step % 4 != 0:
        raise Exception("Step size must be a factor of 4")
    numSamplesPerSegment = segmentTime * X_HZ # Sensor readings per segment for 1 channel
    firstX, firstY = findSegmentStart(data, segmentTime)
    lastX, lastY = findSegmentEnd2(data, segmentTime)
    ySegmentI = firstY
    segments = []
    # Cut up data into segments centered around the label timestamps
    count = 0
    for i in range(firstX, lastX, step):
        segment = {}
        start = i
        end = start + numSamplesPerSegment
        segment["xtime"] = data["xtime"][start:end]
        segLen = len(data["ax"][start:end])
        if segLen < numSamplesPerSegment:
            break
        segment["ax"] = data["ax"][start:end]
        segment["ay"] = data["ay"][start:end]
        segment["az"] = data["az"][start:end]
        segment["gx"] = data["gx"][start:end]
        segment["gy"] = data["gy"][start:end]
        segment["gz"] = data["gz"][start:end]
        segment["ytime"] = [data["ytime"][ySegmentI]]
        if not skipY:
            segment["y"] = [data["y"][ySegmentI]]
        segments.append(segment)
        ySegmentI += int(step/4)
        count += 1

    return segments

def writeFile(path, set, dataType):
    print(path)
    if dataType is "xtime":
        pass
    with open(path, "w") as f:
        for segments in set:
            for segment in segments:
                line = str(segment[dataType][0])
                for point in segment[dataType][1:]: # Write up to the last point
                    line += "," + str(point)
                f.write(line + "\n") # last point

def makeDataDir(folder):
    data = os.curdir + "/data/"
    if not os.path.exists(data):
        os.mkdir(data)
    data = data + folder
    if not os.path.exists(data):
        os.mkdir(data)

def writeFiles(trainSet, testSet):
    folder = "data/train/"
    makeDataDir("train/")
    writeFile(folder + "xtime.csv", trainSet, "xtime")
    writeFile(folder + "ax.csv", trainSet, "ax")
    writeFile(folder + "ay.csv", trainSet, "ay")
    writeFile(folder + "az.csv", trainSet, "az")
    writeFile(folder + "gx.csv", trainSet, "gx")
    writeFile(folder + "gy.csv", trainSet, "gy")
    writeFile(folder + "gz.csv", trainSet, "gz")
    writeFile(folder + "ytime.csv", trainSet, "ytime")
    writeFile(folder + "y.csv", trainSet, "y")

    folder = "data/test/"
    makeDataDir("test/")
    writeFile(folder + "xtime.csv", testSet, "xtime")
    writeFile(folder + "ax.csv", testSet, "ax")
    writeFile(folder + "ay.csv", testSet, "ay")
    writeFile(folder + "az.csv", testSet, "az")
    writeFile(folder + "gx.csv", testSet, "gx")
    writeFile(folder + "gy.csv", testSet, "gy")
    writeFile(folder + "gz.csv", testSet, "gz")
    writeFile(folder + "ytime.csv", testSet, "ytime")
    writeFile(folder + "y.csv", testSet, "y")

def loadData():
    trainFolder = "TrainingData/"
    segmentTime = 1  # 1 Second
    step = 4  # Number of samples to skip before creating a new segment
    segSizes = nList(len(SUBS))
    allSegments = nList(len(SUBS))
    for i in range(len(SUBS)):
        subject = i + 1
        for j in range(len(SUBS[i])):
            session = j + 1
            data = readData(trainFolder, subject, session)
            data = normailizeData(data)
            segments = getSegments(data, segmentTime, step)
            numSegments = len(segments)
            print(str(numSegments) + " segments from Subject: " + str(subject) + "   Session: " + str(session))
            segSizes[i].append(numSegments)
            allSegments[i].append(segments)
    plotData(data, 1,1)
    plt.show()

    testSessionsPerSubject = [2, 2, 1, 0, 1, 1, 2, 0]  # How many sessions to use for the test set per subject
    testSegmentI = []
    for i in range(len(SUBS)):  # Randomly sample the sessions to use as test sets
        index = sorted(random.sample(list(range(0, len(SUBS[i]))), testSessionsPerSubject[i]))
        testSegmentI.append(index)
        print(i, index)
    print("Indicies used for testing:", testSegmentI)

    # Split Train and Test Set
    trainSet = []
    testSet = []
    for i in range(len(testSegmentI)):
        count = 0
        print(i)
        for j in (testSegmentI[i]):
            segments = allSegments[i].pop(j - count)
            testSet.append(segments)
            count += 1
        numRemaining = len(allSegments[i])
        for j in range(numRemaining):
            segments = allSegments[i].pop(0)
            trainSet.append(segments)

    return trainSet, testSet


def main():
    trainSet, testSet = loadData()
    writeFiles(trainSet, testSet)
    print("end")










    #plotData(data, subject, session)
    #plt.show()





if __name__ == '__main__':
    main()