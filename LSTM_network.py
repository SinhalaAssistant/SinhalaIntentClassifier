import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Masking, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):

    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true, axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)

    # 3. Print accuracy score
    print("Accuracy : " + str(accuracy_score(y_true, y_pred)))

    print("")

    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=5))

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix, classes=classes)


train = np.load('data/Train_LSTM.npy')   # Load training data set
test = np.load('data/LSTM_labels.npy')    # Load training lable set

print(test)

labels = np_utils.to_categorical(test,6)

print(train.shape)
print(test.shape)

# Suffle data set
X, y = shuffle(train, labels)

# Split train and test data sets
X_train, X_test, y_train, y_test = train_test_split(
    train, labels, test_size=0.2, random_state=42)

# Initializing values for the LSTM
hidden_layers = 24
batch_size = 64
epochs = 20
learning_rate = 0.01


# Build LSTM
model = Sequential()
model.add(Masking(mask_value=0, input_shape=X.shape[1:]))
# model.add(Conv1D(filters=13, kernel_size=3, padding='same', activation='relu',input_shape=X.shape[1:]))
# model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(hidden_layers, activation='tanh', return_sequences=False))
# model.add(Dense(20,activation='relu'))
model.add(Dense(6, activation='sigmoid'))

# SGD optimizer
sgd = SGD(lr=learning_rate, momentum=0.5)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

results = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_data=(X_test, y_test), shuffle=True)

print(results.history.keys())

model.evaluate(X_test, y_test, batch_size=batch_size)

#Train data visualization
full_multiclass_report(model,X_train,y_train,[0,1,2,3,4,5])
#Test data visualization
full_multiclass_report(model,X_test,y_test,[0,1,2,3,4,5])
