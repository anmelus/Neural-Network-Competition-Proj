import data as d
import pandas as pd
import random
import os

def findNans():
    folder = "data/train/"
    files = ["ax.csv", "ay.csv", "az.csv", "gx.csv", "gy.csv", "gz.csv"]
    data = pd.read_csv(folder + files[0], header=None)
    for i in range(len(data)):
        row = data.iloc[i]
        check = row.isna().sum()
        if check > 0:
            print(i)

def readFile(path):
    with open(path, "r") as f:
        raw = f.readlines()
        return raw


def readFiles(folder, files):
    files = ["ax.csv", "ay.csv", "az.csv", "gx.csv", "gy.csv", "gz.csv", "y.csv"]
    data = {}
    for file in files:
        fileData = readFile(folder + file)
        data[file] = fileData
    return data

def searchDataByY(data, y):
    rows = []
    for i in range(len(data["y.csv"])):
        a = int(data["y.csv"][i])
        if a == y:
            rows.append(i)
    return rows

def setupResampleDirs():
    base = "data/resampled/"
    if not os.path.exists(base):
        os.mkdir(base)
    if not os.path.exists(base + "train/"):
        os.mkdir(base + "train/")
    if not os.path.exists(base + "test/"):
        os.mkdir(base + "test/")


def generate(fType, target):
    folder = "data/" + fType
    files = ["ax.csv", "ay.csv", "az.csv", "gx.csv", "gy.csv", "gz.csv", "y.csv"]
    data = readFiles(folder, files)
    dataRows = [[],[],[],[]]
    for i in range(4):
        rows = searchDataByY(data, i)
        print("Before: ", i, len(rows))
        if len(rows) < target:
            dataRows[i].extend(rows)
            rows = random.choices(rows, k=target - len(rows))
        else:
            rows = random.sample(rows, target)
        dataRows[i].extend(rows)
        print("After: ", i, len(dataRows[i]))
    folder = "data/resampled/" + fType
    setupResampleDirs()
    for f in files:
        with open(folder + f, "w") as out:
            for i in range(len(dataRows)):
                rows = dataRows[i]
                for row in rows:
                    out.write(data[f][row])

def main():
    generate("train/", 15000)
    generate("test/", 5000)










if __name__ == '__main__':
    main()