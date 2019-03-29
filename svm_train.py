from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd

data_new=pd.read_csv("feature.csv")
#data_new=pd.read_csv("features.csv",na_values=['?'])

#data_new.dropna(inplace=True)
features_raw = data_new[['NoduleArea','Perimeter','Diameter' ,'Eccentricity']]
predictions=data_new['prediction']
from sklearn.model_selection import train_test_split

#predict_class = predictions.apply(lambda x: 0 if x == 0 else 1)
np.random.seed(1234)

X_train, X_test, y_train, y_test = train_test_split(features_raw, predictions, train_size=0.90, random_state=1)


# Show the results of the split
print ("Training set has {} samples." .format(X_train.shape[0]))
print ("Testing set has {} samples." .format(X_test.shape[0]))
import sklearn
from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear',C=10,gamma='auto')
svc.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
print(X_test)

predictions_test = svc.predict(X_test)
print(y_test)
predictions_test

