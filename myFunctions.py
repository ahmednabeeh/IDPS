import sys
import numpy as np
import re
import csv
from nltk import word_tokenize
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier

np.set_printoptions(threshold=sys.maxsize)


def cleanWord(token):
    token = token.replace("'", "")
    token = token.replace("www.", "w.")
    token = token.replace("//", "/")
    token = token.lower().replace("  ", " ")
    token = re.sub(r'\d', "0", token.lower().replace("  ", " "))
    token = re.sub(r'(0)\1+', "0", token)
    return token


def sqlKeywordsList():
    with open('sqlKeywords.csv', newline='') as f:
        reader = csv.reader(f)
        sqlKeywordArray = list(reader)

    # Remove [ï»؟a] of the List
    sqlKeywordArray.pop(0)
    exceptionList = []
    sqlKeywordList = []
    for exceptions in sqlKeywordArray:
        for exception in exceptions:
            exceptionString = str(exception)
            # print("Length: ", len(exceptionString), " type: ", type(exceptionString), " ", exceptionString)
            if len(str(exceptionString)) == 4:
                exceptionList.append(exceptionString)
            sqlKeywordList.append(exceptionString)

    dictionary = {
        "exceptionList": exceptionList,
        "sqlKeywordList": sqlKeywordList
    }

    # print(exceptionList)
    # print(sqlKeywordList)
    return dictionary


def buildVocabulary(exceptionList, sqlKeywordList):
    df = pd.read_csv('combinedDatasetModifiedArabic.csv', encoding='utf-8')
    df.shape
    df = df.dropna()
    df = df.sample(frac=1, random_state=1).reset_index()
    label = df['Label'].astype('int')
    sentences = df['Sentence']

    sqlExceptionList = exceptionList
    sqlKeywordList = sqlKeywordList

    # Build Vocabulary
    vocabulary = []
    token4Char = 0
    totalTokens = 0
    for sentence in sentences:
        tokens = word_tokenize(sentence)

        for token in tokens:
            token = cleanWord(token)
            if token:
                if token not in vocabulary:
                    vocabulary.append(token)
                    if len(token) == 4:
                       if token in sqlExceptionList:
                           vocabulary.append(token)
                           token4Char += 1
                           totalTokens += 1
                           # print(token)
                    elif len(token) != 4:
                       vocabulary.append(token)
                       totalTokens += 1

    # Add class_label column to the end of vocabulary to classify bagOfWord
    vocabulary.append("class_label")

    print("----------------------------")
    # Print Lengths
    print("The number of vocabulary items: ", len(vocabulary))
    print("The number of sentences items : ", len(sentences))
    print("Total Tokens = ", totalTokens)
    print("Token 4 Char = ", token4Char)

    print("----------------------------")

    # Build dictionary
    dictionary = {
        "vocabulary": vocabulary,
        "sentences": sentences,
        "label": label
    }
    return dictionary


def bagOfWord(vocabulary, sentences, label, sqlKeywordList, exceptionList):
    # Building BoW array
    indexing = 0
    vectorsArray = np.zeros((len(sentences), len(vocabulary)), dtype='int')
    labelVocabularyIndex = len(vocabulary)

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        for word in tokens:
            word = cleanWord(word)

            # if word and len(word) != 4 and word in exceptionList:
            if word:
                # print("the type = ", type(word), " for: ", word)
                location = vocabulary.index(word.lower())
                wordCount = sentence.lower().count(word.lower())
                # print("word Count in Sentence = ", wordCount)
                # print("**********")
                vectorsArray[indexing, location] = wordCount

        # add the label to the end of vocabulary
        vectorsArray[indexing, labelVocabularyIndex - 1] = label[indexing]
        indexing += 1

    vocabularyLabel = ["words"]
    csvVocabulary = pd.DataFrame(vocabulary, columns=vocabularyLabel)
    csvVocabulary.to_csv("_vocabulary.csv")

    csvDataFrame = pd.DataFrame(vectorsArray, columns=vocabulary)
    csvDataFrame.to_csv("_bagOfWordDatasetArabic.csv")

    return vectorsArray


def buildRequestBoW(requestString):
    df = pd.read_csv('_vocabulary.csv', encoding='utf-8', converters={'words': str})
    vocabulary = df['words'].values.tolist()
    vocabulary.insert(0, "Unnamed: 0")

    # print(len(vocabulary))

    vectorsArray = np.zeros((1, len(vocabulary)), dtype='int')
    labelVocabularyIndex = len(vocabulary)

    sentences = requestString
    sentenceWithoutSplit = sentences
    sentences = sentences.split()

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        # tokens = sentence.split()
        for word in tokens:
            word = cleanWord(word)
            # print(word)

            # if word and len(word) != 4 and word in exceptionList:
            if word:
                # print("the type = ", type(word), " for: ", word)
                location = vocabulary.index(word.lower())
                wordCount = sentenceWithoutSplit.lower().count(word.lower())
                # print("word Count in Sentence = ", wordCount)
                # print("**********")
                vectorsArray[0, location] = wordCount

        # add the label to the end of vocabulary
        vectorsArray[0, labelVocabularyIndex - 1] = 0

    # csvDataFrame = pd.DataFrame(vectorsArray, columns=vocabulary)
    # csvDataFrame.to_csv("requestBoW.csv")
    simpleList = []
    for x in vectorsArray:
        simpleList.append(x)

    # bagOfWord = [vocabulary, simpleList]
    return vocabulary, simpleList


def readVocab():
    df = pd.read_csv('_vocabulary.csv', encoding='utf-8')
    df = df.dropna()
    df = df.sample(frac=1, random_state=1).reset_index()
    sentence = df['words']

    vocabulary = df.values.tolist()

    print(len(vocabulary))

    vectorsArray = np.zeros((1, len(vocabulary)), dtype=int)
    indexing = 0
    labelVocabularyIndex = len(vocabulary)
    for word in sentence:
        word = cleanWord(word)
        if word:
            location = vocabulary.index(word.lower())
            wordCount = sentence.lower().count(word.lower())
            # print("word Count in Sentence = ", wordCount)
            # print("**********")
            vectorsArray[indexing, location] = wordCount

        # add the label to the end of vocabulary
        # vectorsArray[indexing, labelVocabularyIndex - 1] = label[indexing]

        vocabulary.append("class_label")
        print(vectorsArray)


def cvFold():
    df = pd.read_csv("_bagOfWordDatasetArabic.csv")
    y = df["class_label"]
    del df['class_label']
    # Rebuild dataframe
    df.shape
    x = df.loc[:, :]

    # Decision Tree
    classifier = DecisionTreeClassifier(random_state=42)
    kFolds = KFold(n_splits=10)

    accuracy = cross_val_score(classifier, x, y, cv=kFolds)

    print("Cross Validation Accuracy of 10 K for Decision Tree: ", accuracy)
    print("Cross Validation Average of 10 K for Decision Tree: ", accuracy.mean())
    # print("Number of CV Scores used in Average: ", len(accuracy))
    print("---------------------------")
