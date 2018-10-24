import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.utils  import class_weight
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping

import talos as ta

dataframe = pandas.read_csv("NewMFCCFeaturesWpadding.csv", header=None)   #data/NewMFCCFeaturesWpadding
dataset = dataframe.values
np.random.shuffle(dataset)
# print(dataset)


print(dataset.shape)
X = dataset[:,0:12337].astype(float)  #186576  7150
Y = dataset[:,12337]


labels = np_utils.to_categorical(Y,6)

print(X.shape)
print(labels.shape)

class_weights = class_weight.compute_class_weight('balanced',np.unique(Y),Y)

def best_model(x_train,x_test,y_train,y_test,params):
    global class_weights

    model = Sequential()
    model.add(Dense(params['first_neuron'] ,input_dim=x_train.shape[1], activation=params['activation'],kernel_initializer=params['kernel_initializer'])) #500   #kernel_regularizer=regularizers.l1(0.001)
    model.add(Dropout(params['dropout']))                      #0.4
    model.add(Dense(500,activation=params['activation'], kernel_regularizer=regularizers.l2(0.01), kernel_initializer=params['kernel_initializer']))      #this is not here l2 500
    model.add(Dropout(params['dropout']))                       #this is not there
    model.add(Dense(500,activation=params['activation'],kernel_regularizer=regularizers.l2(0.01), kernel_initializer=params['kernel_initializer']))     #200
    model.add(Dropout(params['dropout']))                             #this is not there
    model.add(Dense(500,activation=params['activation'],kernel_regularizer=regularizers.l2(0.01), kernel_initializer=params['kernel_initializer']))   #100
    model.add(Dropout(params['dropout']))                     #0.25                                                                                       0   
    model.add(Dense(100,activation=params['activation'],kernel_regularizer=regularizers.l2(0.01), kernel_initializer=params['kernel_initializer']))  #80
    model.add(Dropout(params['dropout'])) 
    model.add(Dense(y_train.shape[1],activation='softmax'), kernel_initializer=params['kernel_initializer'])    #6

    sgd = SGD(lr=params['lr'], decay=1e-6, momentum=params['momentum'], nesterov=True)

    model.compile(loss=params['losses'], optimizer=sgd, metrics=['accuracy'])

    history = model.fit(x_train,y_train,batch_size=params['batch_size'] , epochs=params['epochs'],class_weight=class_weights, validation_data=(x_test,y_test), verbose=0) #64

    return history, model

p = {'first_neuron':[9,10,11],
     'hidden_layers':[0, 1, 2],
     'batch_size': [32,64,128,256],
     'epochs': [100,200,300],
     'dropout': [0,0.25,0.4,0.5],
     'kernel_initializer': ['','normal'],
     'momentum': [0.6,0.7,0.8,0.9],
     'losses': [binary_crossentropy],
     'activation':[relu, elu],
     'lr': [0.1,0.01,0.001]}


t = ta.Scan(x=X,
            y=labels,
            model=best_model,
            grid_downsample=0.05,
            params=p,
            dataset_name='speech recognition',
            experiment_no='1')