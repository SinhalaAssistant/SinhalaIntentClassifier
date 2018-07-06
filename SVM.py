import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd 

print('Initializing imports....')

print('Loading dataset...')
#Loading dataset from 'data/NewMFCCFeaturesWpadding.csv'

dataframe = pd.read_csv('data/NewMFCCFeaturesWpadding.csv')
dataset = dataframe.values
np.random.shuffle(dataset)

print('Loading data complete')

#Spliting dataset

X = dataset[:,0:12337].astype(float)
Y = dataset[:,12337]

print('traning data shape : ')
print(X.shape)
print('Lable data shape :')
print(Y.shape)

#Spliting data to train and test set

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

print('X_train data shape : ')
print(X_train.shape)
print('X_test data shape : ')
print(X_test.shape)
print('y_train data shape : ')
print(y_train.shape)
print('y_test data shape : ')
print(y_test.shape)

print('Building the SVM model....')

model = svm.SVC(kernel='linear',C=1)

print('Training the model...')

model.fit(X_train,y_train)

print('Training is complete')

score = model.score(X_test,y_test)

print('model score : ')
print(score)

