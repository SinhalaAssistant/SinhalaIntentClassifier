import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def create_model():
    # Define layer nodes
    input_nodes = 500
    hidden_layer_1 = 200
    hidden_layer_2 = 100
    hidden_layer_3 = 80
    output_layer = 6

    dropout = 0.2

    # Define loss function
    loss_function = 'catalorical_crossentropy'

    # Define optimiser
    learning_rate = 0.01
    momentum = 0.9

    model = Sequential()
    # 500   #kernel_regularizer=regularizers.l2(0.1),
    model.add(Dense(input_nodes, input_dim=12337, activation='relu'))
    model.add(Dropout(0.5))  # 0.5
    model.add(Dense(hidden_layer_1, activation='relu'))  # 200 
    model.add(Dense(hidden_layer_2, activation='relu'))  # 100
    model.add(Dropout(0.25))  # 0.25
    model.add(Dense(hidden_layer_3, activation='relu'))  # 80
    model.add(Dense(output_layer, activation='softmax'))  # 6

    optimizer = SGD(lr=learning_rate, momentum=momentum)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    return model

#load data
dataframe = pd.read_csv("data/NewMFCCFeaturesWpadding.csv", header=None)
dataset = dataframe.values
np.random.shuffle(dataset)

#split dataset data and its labels
print(dataset.shape)
X = dataset[:,0:12337].astype(float)  #186576  7150
Y = dataset[:,12337]

labels = np_utils.to_categorical(Y,6)

model = KerasClassifier(build_fn=create_model)

epochs = [50,100,150,200,250]
batch_size = [32,64,128,256]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model,param_grid=param_grid, n_jobs=-1)

grid_result = grid.fit(X,Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))