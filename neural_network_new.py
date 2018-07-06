import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

X = np.load('training.npy')
y = np_utils.to_categorical(np.load('test.npy'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def create_model():

    model = Sequential()
    model.add(Dense(150, input_dim=186576, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model

# model.summary()

result = model.fit(X_train, y_train, batch_size=512,
                   epochs=150, shuffle=True, validation_data=(X_test, y_test))


# print(np.mean(result.history["val_acc"]))
