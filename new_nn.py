import pandas
import numpy as np
import itertools
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import regularizers
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.optimizers import SGD

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

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
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)



# seed = 7
# np.random.seed(seed)

	
# load dataset
dataframe = pandas.read_csv("data/mfcc_new.csv", header=None)
dataset = dataframe.values
np.random.shuffle(dataset)
# print(dataset)

print(dataset.shape)
X = dataset[:,0:12337].astype(float)  #186576  7150
Y = dataset[:,12337]

# train_x = X[:320]
# test_x = X[320:]

# train = np.load('data/Train_LSTM.npy')   # Load training data set
# test = np.load('data/LSTM_labels.npy')    # Load training lable set

labels = np_utils.to_categorical(Y,6)

print(X.shape)
print(labels.shape)

# Suffle data set
# X, y = shuffle(train, labels)

# Split train and test data sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42)

# print(dataframe)
# print(Y)
# encoder = LabelEncoder()

# # print(X)

# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = np_utils.to_categorical(encoded_Y)


model = Sequential()
model.add(Dense(500 ,input_dim=12337,kernel_regularizer=regularizers.l2(0.01), activation='relu'))   #500
model.add(Dropout(0.6))
model.add(Dense(300,activation='relu'))      #200
model.add(Dropout(0.5))
model.add(Dense(200,activation='relu'))      #200
model.add(Dropout(0.4))
model.add(Dense(100,activation='relu'))     #100
model.add(Dropout(0.25))                    #0.25
model.add(Dense(80,activation='relu'))      #80
model.add(Dense(6,activation='softmax'))    #6

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()

results = model.fit(X_train,y_train,batch_size=64, epochs=200, validation_data=(X_test,y_test)); #64

# model.evaluate(X,dummy_y,batch_size=5)

# print(model.predict(test_x))
# print(test_y)

plot_history(results)

# score = model.evaluate(test_x,test_y);

full_multiclass_report(model,X_train,y_train,[0,1,2,3,4,5])

full_multiclass_report(model,X_test,y_test,[0,1,2,3,4,5])

model.save('new_model.h5')

# his = model.evaluate(X_train,batch_size=64)

# print(his)

# print(score)
# print(model.history)
print(np.mean(results.history["val_acc"]))