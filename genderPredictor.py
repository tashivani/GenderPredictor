#!/usr/bin/python

import re
import csv
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pandas import DataFrame

FEMALE = 0
MALE = 1
countVec = CountVectorizer()
classifier = MultinomialNB()


def main():
    filename = 'blog-gender-dataset.csv'
    dataSet = loadCsv(filename)
    splitRatio = 0.70
    splitSize = int(len(dataSet) * splitRatio)

    train_dataSet = dataSet[:splitSize]
    test_dataSet = dataSet[splitSize:]

    trainModelWith(train_dataSet)
    predictUsing(test_dataSet)


def trainModelWith(train_dataSet):
    print "> Training the Naive Bayes model on training data set..."

    rows = []
    index = []
    count = 0

    for data in train_dataSet:
        rows.append({'text': data[1], 'gender': data[0]})
        index.append(count)
        count+=1

    dataFrame = DataFrame(rows, index=index)
    beginTime = time.clock()

    counts = countVec.fit_transform(dataFrame['text'].values) #does two things: it learns the vocabulary of the corpus and extracts word count features.
    targets = dataFrame['gender'].values
    classifier.fit(counts, targets)

    endTime = time.clock()
    print "> Training completed in {0:.3f} ms.\n".format((endTime-beginTime))


def predictUsing(test_dataSet):
    print "> Performing gender classification on testing data..."

    texts = []
    actualGender = []

    for i in range(len(test_dataSet)):
        actualGender.append(test_dataSet[i][0])
        texts.append(test_dataSet[i][1])

    beginTime = time.clock()
    test_counts = countVec.transform(texts)
    probability = classifier.predict_proba(test_counts)
    endTime = time.clock()

    print "> Prediction completed in {0:.3f} ms.\n".format((endTime-beginTime))

    printResults(probability, texts, actualGender)


def printResults(probability, textArr, actualGenderArr):
    print "> Printing result..."

    gender = {
        FEMALE: 'Female',
        MALE: 'Male'
    }
    successes = 0
    result = "\ntPREDICTED_GENDER: PROBABILITY, ACTUAL_GENDER, TEXT\n"

    for i in range(len(textArr)):
        predictedGender = FEMALE
        if probability[i][FEMALE] < probability[i][MALE]:
            predictedGender = MALE

        if predictedGender == actualGenderArr[i]:
            successes += 1

        result += "\n{0}:{1:.2f}%, {2}, {3}".format(gender[predictedGender], probability[i][predictedGender] * 100, gender[actualGenderArr[i]], textArr[i])

    total = len(textArr)
    accuracy = float(successes) / total * 100.00
    print "\n# Predictions:\t\t\t\t{0}\n# Successful predictions:\t{1}\nAccuracy:\t\t\t\t\t{2}%".format(total, successes, round(accuracy, 2))
    text_file = open("Output.txt", "w")
    text_file.write(result)
    text_file.close()


def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataSet = list(lines)
    newDataSet = []
    for i in range(len(dataSet)):
        if dataSet[i]:
            gender = dataSet[i][0].strip()
            gender = gender.lower()

            if gender == 'm' or gender == 'f':
                genderInt = FEMALE

                if gender == 'm':
                    genderInt = MALE

                str = dataSet[i][1]
                str = str.lower()
                str = re.sub('\\n', ' ', str)
                str = re.sub('\\t', ' ', str)
                str = re.sub('\s+', ' ', str)
                row = [genderInt, str]
                newDataSet.append(row)

    return newDataSet

main()