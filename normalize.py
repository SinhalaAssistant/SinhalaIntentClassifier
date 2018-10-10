import numpy as np
import pandas as pd 
from sklearn import preprocessing


dataframe = pd.read_csv("data/NewMFCCFeaturesWpadding.csv", header=None)
dataset = dataframe.values
# np.random.shuffle(dataset)
# print(dataset)

print(dataset.shape)
X = dataset[:,0:12337].astype(float)  #186576  7150
Y = dataset[:,12337]

print(X.shape)
print(Y.shape)
print(X)
print(Y)

transposed = X.transpose()

print(transposed)

print("============================================================")

max_abs_scaler = preprocessing.MaxAbsScaler()

normalized_transposed_x = max_abs_scaler.fit_transform(transposed)

print(normalized_transposed_x)
print(normalized_transposed_x.shape)

normalized_x = normalized_transposed_x.transpose()

print("============================================================")

Y = Y.reshape((Y.shape[0],1))
print(normalized_x.shape)
print(Y.shape) 



normalized_x = np.concatenate((normalized_x, Y), axis=1)

print(normalized_x)


np.savetxt('normalized_MFCC.csv', normalized_x, delimiter=',')

