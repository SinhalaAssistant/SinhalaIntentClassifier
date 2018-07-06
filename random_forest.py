import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.externals import joblib

print('Libraries imported.....')

dataframe = pd.read_csv("data/NewMFCCFeaturesWpadding.csv", header=None)
dataset = dataframe.values
np.random.shuffle(dataset)
# print(dataset)

print(dataset.shape)
X = dataset[:,0:12337].astype(float)  #186576  7150
Y = dataset[:,12337]

print(X.shape)
print(Y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

classifire = RandomForestClassifier(n_estimators=1500,criterion='gini')
classifire.fit(X_train,y_train)

result = classifire.predict(X_test)
print(y_test)
print(result)

cm = confusion_matrix(y_test,result)

print precision_score(y_test, result, average='macro')

print(cm)