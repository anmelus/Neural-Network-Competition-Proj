import data as d
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU usage
import matplotlib.pyplot as plt
import seaborn as sn
import pretty

SUB9 = ["009", "01"]
SUB10 = ["010", "01"]
SUB11 = ["011", "01"]
SUB12 = ["012", "01"]
SUBS = [SUB9, SUB10, SUB11, SUB12]

def readMyCSV(path):
    data = []
    with open(path, "r") as f:
        raw = f.readlines()
        for line in raw:
            line = line.split(",")
            data.append(int(line[0]))
    return data

def readSequence(folder, subject, session):
    dataPath = folder + "subject_" + str(subject) + "_" + str(session)
    data = {}  # Dictionary to hold all data
    # Read X Time
    tmp = d.readCSV(dataPath + "__x_time.csv", float)
    data["xtime"] = tmp[0]
    # Read X Data
    tmp = d.readCSV(dataPath + "__x.csv", float)
    data["ax"] = tmp[0]
    data["ay"] = tmp[1]
    data["az"] = tmp[2]
    data["gx"] = tmp[3]
    data["gy"] = tmp[4]
    data["gz"] = tmp[5]
    # Read Y Time
    tmp = d.readCSV(dataPath + "__y_time.csv", float)
    data["ytime"] = tmp[0]
    # Read Y Data
    return data


def plot_loss(history, lossOnly=False):
    plt.figure()
    plt.title("Loss")
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')

    plt.figure()
    plt.title("Accuracy")
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    #plt.ylabel('Error')
    plt.legend()

def readFiles(folder, files):
    data = []
    for file in files:
        fileData=pd.read_csv(folder + file, header=None)
        #fileData.dropna(inplace=True)
        check = fileData.isna().sum()
        if sum(check) > 0:
            raise Exception("NANs in data")
        data.append(fileData.values)
    return np.dstack(data)

def readData(folder):
    files = ["ax.csv","ay.csv","az.csv","gx.csv","gy.csv","gz.csv"]
    x = readFiles(folder, files)
    y = pd.read_csv(folder + "y.csv", header=None)
    return x, y

def fitModel(model, trainX, trainY, testX, testY, epochs, batchSize, verbose,enableEarlyStopping=True):
    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=.0001),
                  metrics=['acc'])
    if enableEarlyStopping:
        earlystopping = tf.keras.callbacks.EarlyStopping(
            # monitor="val_loss",
            monitor="val_acc",
            patience=5,
            restore_best_weights=True,
            verbose=1)
        history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, verbose=verbose,
                            validation_data=(testX, testY), callbacks=[earlystopping])
    else:
        history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, verbose=verbose,
                            validation_data=(testX, testY))
    return model, history


def createModel(trainX, trainY, testX, testY):
    verbose = 1
    epochs = 100
    batchSize = 64
    _, timesteps, channels = trainX.shape
    _, numCategories = trainY.shape
    model = tf.keras.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(timesteps, channels)),
        layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
        layers.Dropout(.5),
        layers.MaxPool1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(numCategories, activation="softmax")
    ])
    enableEarlyStopping = True
    model, history = fitModel(model, trainX, trainY, testX, testY, epochs, batchSize, verbose,enableEarlyStopping)
    plot_loss(history)
    # evaluate model
    _, accuracy = model.evaluate(testX, testY, batch_size=batchSize, verbose=verbose)

    return model

def confusionPlot(y, pred):
    confusion = tf.math.confusion_matrix(y, pred, num_classes=4)
    confusionArray = confusion.numpy()
    print(confusion)
    df_cm = pd.DataFrame(confusionArray, ["solid ground", "downstairs", "upstairs", "grass ground"],
                         ["solid ground", "downstairs", "upstairs", "grass ground"])
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm, annot=True)  # font size
    plt.xlim([0, 4])
    plt.ylim([4, 0])

    pretty.pretty_plot_confusion_matrix(df_cm, cmap="PuRd")
    plt.xlim([0, 5])
    plt.ylim([5, 0])



def segmentToModelInput(segments):
    data = [[],[],[],[],[],[]]
    for segment in segments:
        data[0].append(segment["ax"])
        data[1].append(segment["ay"])
        data[2].append(segment["az"])
        data[3].append(segment["gx"])
        data[4].append(segment["gy"])
        data[5].append(segment["gz"])
    df = []
    for channel in data:
        chan = pd.DataFrame(channel).values
        df.append(chan)
    return np.dstack(df)

def transformSequenceIntoSegments(folder, step, segTime, sub, seq):

    data = readSequence(folder, sub, seq)
    data = d.normailizeData(data)
    startX, startY = d.findSegmentStart(data, segTime)
    endX, endY = d.findSegmentEnd2(data, segTime)
    remainingYIStart = len(list(range(0,startY)))

    remainingYIEnd = list(range(endY + 1, len(data["ytime"])))
    segments = d.getSegments(data, segTime, step, True)
    remainingYIEnd = len(data["ytime"]) - remainingYIStart - len(segments)
    totalSegmentsNeeded = len(data["ytime"])
    numSegments = len(segments)
    print(startX, startY)
    print(endX, endY)
    print(remainingYIStart)
    print(remainingYIEnd)
    return startY, endY, remainingYIStart, remainingYIEnd, segments

def predictSequence(model, step, segTime, folder, sub, session):
        startY, endY, remainingYIStart, remainingYIEnd, segments = transformSequenceIntoSegments(folder, step, segTime, sub, session)
        testInput = segmentToModelInput(segments)
        testPred = model.predict(testInput)
        testPredArray = list(np.argmax(testPred, axis=1))
        firstPred = testPredArray[0]
        for i in range(remainingYIStart):
            testPredArray.insert(0,firstPred)
        lastPred = testPredArray[-1]
        for i in range(remainingYIEnd):
            testPredArray.insert(-1,lastPred)
        return testPredArray


def runPredictions(model, step, segTime):
    folder = "TestData/"
    for sub in SUBS:
        sub = sub[0]
        session = "01"
        testPredArray = predictSequence(model, step, segTime, folder, sub, session)
        with open("predictions/subject_" + sub + '_' + session + "__y.csv", "w") as f:
            for point in testPredArray:
                f.write(str(point) + "\n")


def build():
    trainFolder = "data/resampled/train/"
    testFolder = "data/resampled/test/"
    #testFolder = "data/test/"
    trainX, trainY = readData(trainFolder)

    trainYOneHot = tf.keras.utils.to_categorical(trainY)
    testX, testY = readData(testFolder)
    testYOneHot = tf.keras.utils.to_categorical(testY)
    model = createModel(trainX, trainYOneHot, testX, testYOneHot)
    testPred = model.predict(testX)
    testPredArray = np.argmax(testPred, axis=1)
    confusionPlot(testY, testPredArray)
    model.save('my_model.h5')

    plt.show()
    print()


def run():
    step = 4
    segTime = 1
    modelPath = 'my_model.h5'
    model = tf.keras.models.load_model(modelPath)
    model.summary()

    testX, testY = readData("data/resampled/test/")
    testYOneHot = tf.keras.utils.to_categorical(testY)
    loss, acc = model.evaluate(testX, testYOneHot, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    #runPredictions(model, step, segTime)
    pred = predictSequence(model, step, segTime, "TrainingData/", "001", "06")
    true = readMyCSV("TrainingData/subject_001_06__y.csv")
    confusionPlot(true, pred)
    plt.show()
    print()



if __name__ == '__main__':
    build()
    #run()