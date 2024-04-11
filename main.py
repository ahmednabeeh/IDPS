from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from joblib import dump, load
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import psutil
import time
import os


csvColumn = ["Algorithm", "Test Size", "Round", "Accuracy", "Testing Time", "Training Time", "Precision", "Recall",
             "F1score", "Confusion Matrix", "True Positive Rate", "False Positive Rate", "AUC", "Build Model Size",
             "Memory Size", "Execution Time"]

df = pd.read_csv("_bagOfWordDataset.csv", encoding='utf8')
df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
df.shape

y = df["class_label"]
del df['class_label']

# Must be dropt in XGBoost
del df['[']
del df[']']
del df['<']
# Rebuild dataframe

df.shape
x = df.loc[:, :]

resultList = []
scriptTime = time.time()

for testSize in range(6, 9):

    testSizeValue = "0." + str(testSize)
    testSizeValue = float(testSizeValue)
    print("the type: ", type, " - ", testSizeValue)

    for testRound in range(1, 11):

        model = None
        auc = None
        fpr = None
        tpr = None
        splitTime = time.time()

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSizeValue, random_state=1)
        splitTime = "%s" % (time.time() - splitTime)
        print("Split Time", splitTime)
        print("Lap Value: ", testSize, testRound, testSizeValue)
        start_time = time.time()

        '''
        ----------------------------------------------------
        '''

        start_time = time.time()
        memorySize = 0
        algorithmName = "xGBoost"
        print(algorithmName)

        model = None
        auc = None
        fpr = None
        tpr = None

        trainTime = time.time()
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"
        dump(model, modelName)
        file_size = round(os.path.getsize(modelName) / 1024)
        file_size = str(file_size) + "KB"

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()

        precision = precision_score(y_test, predicted_y, average='micro')
        recall = recall_score(y_test, predicted_y, average='micro')
        f1score = f1_score(y_test, predicted_y, average='micro')

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        print("Y Proba, ", y_pred_proba, " Len proba: ", len(y_test))
        print("Y test, ", y_test, " Len y: ", len(y_test))
        # _ represent the thresh hold

        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)

        roc_auc_scores = []
        for i in range(2):
            y_true = [1 if x == i else 0 for x in y_test]
            y_pred = model.predict_proba(X_test)[:, i]
            roc_auc_scores.append(roc_auc_score(y_true, y_pred))

        # Take unweighted average of ROC AUC scores
        multi_class_roc_auc_score = sum(roc_auc_scores) / len(roc_auc_scores)

        print("Multi Class AUC = ", multi_class_roc_auc_score)

        auc = multi_class_roc_auc_score
        xGBoostAUC = auc

        print(auc)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(xGBoostAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        executionTime = "%s" % (time.time() - start_time)
        memorySize = str(round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)) + "MB"
        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, xGBoostAUC, file_size, memorySize, executionTime])
        memorySize = 0

        print(" ")
        print(" ------------------------------- ")
        print(" ")

        '''
        ----------------------------------------------------
        '''

        start_time = time.time()
        memorySize = 0
        algorithmName = "adaBoost"
        print(algorithmName)

        model = None
        auc = None
        fpr = None
        tpr = None

        trainTime = time.time()
        model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"
        dump(model, modelName)
        file_size = round(os.path.getsize(modelName) / 1024)
        file_size = str(file_size) + "KB"

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()

        precision = precision_score(y_test, predicted_y, average='micro')
        recall = recall_score(y_test, predicted_y, average='micro')
        f1score = f1_score(y_test, predicted_y, average='micro')

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        print("Y Proba, ", y_pred_proba, " Len proba: ", len(y_test))
        print("Y test, ", y_test, " Len y: ", len(y_test))
        # _ represent the thresh hold

        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)

        roc_auc_scores = []
        for i in range(2):
            y_true = [1 if x == i else 0 for x in y_test]
            y_pred = model.predict_proba(X_test)[:, i]
            roc_auc_scores.append(roc_auc_score(y_true, y_pred))

        # Take unweighted average of ROC AUC scores
        multi_class_roc_auc_score = sum(roc_auc_scores) / len(roc_auc_scores)

        print("Multi Class AUC= ", multi_class_roc_auc_score)

        auc = multi_class_roc_auc_score
        adaBoostAUC = auc

        print(auc)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(adaBoostAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        executionTime = "%s" % (time.time() - start_time)
        memorySize = str(round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)) + "MB"
        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, adaBoostAUC, file_size, memorySize, executionTime])
        memorySize = 0

        print(" ")
        print(" ------------------------------- ")
        print(" ")

        '''
        ----------------------------------------------------
        '''
        start_time = time.time()
        memorySize = 0
        algorithmName = "randomForest"
        print(algorithmName)

        model = None
        auc = None
        fpr = None
        tpr = None

        trainTime = time.time()
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"
        dump(model, modelName)
        file_size = round(os.path.getsize(modelName) / 1024)
        file_size = str(file_size) + "KB"

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()

        precision = precision_score(y_test, predicted_y, average='micro')
        recall = recall_score(y_test, predicted_y, average='micro')
        f1score = f1_score(y_test, predicted_y, average='micro')

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        print("Y Proba, ", y_pred_proba, " Len proba: ", len(y_test))
        print("Y test, ", y_test, " Len y: ", len(y_test))
        # _ represent the thresh hold

        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)

        roc_auc_scores = []
        for i in range(2):
            y_true = [1 if x == i else 0 for x in y_test]
            y_pred = model.predict_proba(X_test)[:, i]
            roc_auc_scores.append(roc_auc_score(y_true, y_pred))

        # Take unweighted average of ROC AUC scores
        multi_class_roc_auc_score = sum(roc_auc_scores) / len(roc_auc_scores)

        print("Multi Class AUC= ", multi_class_roc_auc_score)

        auc = multi_class_roc_auc_score
        randomForestAUC = auc

        print(auc)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(randomForestAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        executionTime = "%s" % (time.time() - start_time)
        memorySize = str(round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)) + "MB"
        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, randomForestAUC, file_size, memorySize, executionTime])
        memorySize = 0

        print(" ")
        print(" ------------------------------- ")
        print(" ")

        '''
        ----------------------------------------------------
        '''

        start_time = time.time()
        memorySize = 0
        algorithmName = "naiveBayes"
        print(algorithmName)

        model = None
        auc = None
        fpr = None
        tpr = None

        trainTime = time.time()
        model = GaussianNB()
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"
        dump(model, modelName)
        file_size = round(os.path.getsize(modelName) / 1024)
        file_size = str(file_size) + "KB"

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()

        precision = precision_score(y_test, predicted_y, average='micro')
        recall = recall_score(y_test, predicted_y, average='micro')
        f1score = f1_score(y_test, predicted_y, average='micro')

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        print("Y Proba, ", y_pred_proba, " Len proba: ", len(y_test))
        print("Y test, ", y_test, " Len y: ", len(y_test))
        # _ represent the thresh hold

        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)

        roc_auc_scores = []
        for i in range(2):
            y_true = [1 if x == i else 0 for x in y_test]
            y_pred = model.predict_proba(X_test)[:, i]
            roc_auc_scores.append(roc_auc_score(y_true, y_pred))

        # Take unweighted average of ROC AUC scores
        multi_class_roc_auc_score = sum(roc_auc_scores) / len(roc_auc_scores)

        print("Multi Class AUC= ", multi_class_roc_auc_score)

        auc = multi_class_roc_auc_score
        naiveBayesAUC = auc

        print(auc)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(naiveBayesAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        executionTime = "%s" % (time.time() - start_time)
        memorySize = str(round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)) + "MB"
        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, naiveBayesAUC, file_size, memorySize, executionTime])
        memorySize = 0

        print(" ")
        print(" ------------------------------- ")
        print(" ")

        '''
        ----------------------------------------------------
        '''

        start_time = time.time()
        memorySize = 0
        algorithmName = "decisionTree"
        print(algorithmName)

        model = None
        auc = None
        fpr = None
        tpr = None

        trainTime = time.time()
        model = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"
        dump(model, modelName)
        file_size = round(os.path.getsize(modelName) / 1024)
        file_size = str(file_size) + "KB"

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()

        precision = precision_score(y_test, predicted_y, average='micro')
        recall = recall_score(y_test, predicted_y, average='micro')
        f1score = f1_score(y_test, predicted_y, average='micro')

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        print("Y Proba, ", y_pred_proba, " Len proba: ", len(y_test))
        print("Y test, ", y_test, " Len y: ", len(y_test))
        # _ represent the thresh hold

        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)

        roc_auc_scores = []
        for i in range(2):
            y_true = [1 if x == i else 0 for x in y_test]
            y_pred = model.predict_proba(X_test)[:, i]
            roc_auc_scores.append(roc_auc_score(y_true, y_pred))

        # Take unweighted average of ROC AUC scores
        multi_class_roc_auc_score = sum(roc_auc_scores) / len(roc_auc_scores)

        print("Multi Class AUC= ", multi_class_roc_auc_score)

        auc = multi_class_roc_auc_score
        decisionTreeAUC = auc

        print(auc)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(decisionTreeAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        executionTime = "%s" % (time.time() - start_time)
        memorySize = str(round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)) + "MB"
        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, decisionTreeAUC, file_size, memorySize, executionTime])
        memorySize = 0

        print(" ")
        print(" ------------------------------- ")
        print(" ")

        print("%s" % (time.time() - start_time))
        print("-------------------> End of Round")

        fileName = "comparison/comparison-" + str(testSize).zfill(2) + "-" + str(testRound).zfill(2) + ".csv"

        csvDataFrame = pd.DataFrame(resultList, columns=csvColumn)
        csvDataFrame.to_csv(fileName)

# Script execution Time
print("%s" % (time.time() - scriptTime))
