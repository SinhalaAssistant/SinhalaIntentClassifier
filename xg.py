import xgboost as xgb 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import  precision_score

# Loading data set
dataframe = pd.read_csv("data/NewMFCCFeaturesWpadding.csv", header=None)
dataset = dataframe.values
np.random.shuffle(dataset)

print(dataset.shape)
X = dataset[:,0:12337].astype(float)  #186576  7150
Y = dataset[:,12337]

# Spliting test train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train) 
dtest = xgb.DMatrix(X_test, label=y_test)

#Define  params for XGboost

params = {
    'max-depth' : 8,
    'eta' : 0.01,
    'silent' : 0,
    'objective' : 'multi:softprob',
    'num_class' : 6 
}

num_round = 30

# Training the xgb with traing data set
bst = xgb.train(params, dtrain, num_round)

# Predicting the test data sample
preds = bst.predict(dtest)

# Taking the real predicted value
best = np.asarray([np.argmax(line) for line in preds])

print precision_score(y_test, best, average='macro')