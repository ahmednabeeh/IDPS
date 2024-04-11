import time
import pandas as pd
import myFunctions
start_time = time.time()


from joblib import dump, load
# dump(classifier, 'loadModel.joblib')

classifier = load('saved-model/modelName.joblib')

requestBagOfWord = myFunctions.buildRequestBoW("1%\" )) ) and 6537=dbms_pipe.receive_message ( chr ( 76 ) ||chr ( 116 ) ||chr ( 117 ) ||chr ( 65 ) ,5 ) and (( ( \"%\"=\"")

requestDF = pd.DataFrame(requestBagOfWord[1], columns=requestBagOfWord[0])

request_y = requestDF["class_label"]

# Delete label column
del requestDF['class_label']
requestDF.shape

y_pred = classifier.predict(requestDF)
print("The y Predict: ", y_pred)
if 0 in y_pred:
    print("no attack")
elif 1 in y_pred:
    print("SQL Injection")
elif 2 in y_pred:
    print("XSS Attack")

print("--- %s seconds ---" % (time.time() - start_time))
