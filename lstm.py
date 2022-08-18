# lstm model
import shutil
import numpy as np
from numpy import mean
from numpy import newaxis
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, MaxPooling3D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)}, threshold=np.inf, linewidth=np.inf)

# load the dataset, returns train and test X and y elements

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "return_sequences": self.return_sequences,
        })
        return config


def print_confusion_matrix(predict,step,dirname):
    try:
        d2_arr = np.empty((0,3),float)
        for k in range(0, 3):
            temp_arr = np.zeros(3)
            for i in range(k * step, (k + 1) * step):
                temp_arr = temp_arr + predict[i]
            temp_arr = (temp_arr / step) * 100
            temp_arr2 = temp_arr.reshape(1,3)
            d2_arr = np.append(d2_arr, temp_arr2,axis=0)
        print(d2_arr)
        plt.imshow(d2_arr,interpolation='nearest')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.colorbar()
        #plt.show()

        disp = ConfusionMatrixDisplay(confusion_matrix=d2_arr,display_labels=list(['init','attack','youtube']))
        disp.plot(values_format='.2f')

        plt.savefig('./' + dirname + '/cf.png')
    except:
        return

def data_preprocessing():
    os.chdir("forlstm/")
    filenames = os.listdir()
    print(len(filenames))
    raw = pd.read_csv('init.csv')
    # trainX = raw.values[0:70000, 0:32] 2000*35
    # trainy = raw.values[0:350, 32]
    # testX = raw.values[70000:, 0:32]
    # testy = raw.values[350:500, 32]
    #raw.drop(columns=['36','44','60','28','20','52','12','4','26','18'])

    #trainX = raw.values[2000:, 1:55] # 10,800,64
    trainX = raw.loc[2000:, ['15','19','7','3','23','9','5','11','8','6']]
    trainy = raw.values[0:800, 0]
    testX = raw.loc[1:2000, ['15','19','7','3','23','9','5','11','8','6']]
    testy = raw.values[800:1000, 0]
    print('data loding..')
    print(str(0) + ': ', end='')
    print(trainX.shape, end="")
    print(testX.shape)

    for i, filename in enumerate(filenames):
        if i > 0:
            raws = pd.read_csv(filename, thousands=',')
            new_trainX = raws.loc[2000:, ['15','19','7','3','23','9','5','11','8','6']]
            new_trainy = raws.values[0:800, 0]
            new_testX = raws.loc[1:2000, ['15','19','7','3','23','9','5','11','8','6']]
            new_testy = raws.values[800:1000, 0]

            trainX = np.append(trainX, new_trainX, axis=0)
            trainy = np.append(trainy, new_trainy, axis=0)
            testX = np.append(testX, new_testX, axis=0)
            testy = np.append(testy, new_testy, axis=0)

            print(str(i) + ': ',end='')
            print(trainX.shape,end="")
            print(testX.shape,end=' - check: ')
            print(str(i+1)+'*'+str(2000)+'='+str((i+1)*2000))

    os.chdir('../PycharmProjects/ML_for_lab/crpytojacking-detection--main/')

    return trainX, testX, trainy, testy


def load_dataset(prefix=''):
    # load all train

    scaler = StandardScaler()

    trainX, testX, trainy, testy = data_preprocessing()
    print(trainX.shape, trainy.shape)
    print(testX.shape, testy.shape)

    # one hot encode y
    # https://keras.io/ko/losses/
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    print("data nomalization .. (loding..) ")
    trainX = scaler.fit_transform(trainX)
    testX = scaler.fit_transform(testX)

    # 2Dto3D
    trainX = trainX.reshape(2400, 10,10)
    testX = testX.reshape(600, 10, 10)

    # (6000,32,1) & (1500,32,1)
    #    trainX = np.expand_dims(trainX, axis=2)
    #    testX = np.expand_dims(testX, axis=2)

    # trainX = trainX[:,newaxis,:]
    # testX = testX[:,newaxis,:]

    print(trainX.shape, trainy.shape, testX.shape, testy.shape)

    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 100, 10
    n_timesteps, n_features, n_outputs = 10, 10, 3
    # verbose, epochs, batch_size = 1, 150, 10
    # n_timesteps, n_features, n_outputs = 300, 32, 10
    model = Sequential()
    # model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(n_timesteps, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    #model.add(BatchNormalization())
    model.add(Attention(return_sequences=False))
    #model.add(BatchNormalization())
    model.add(Dense(n_outputs, activation='softmax'))
    model.summary()

    es = EarlyStopping(monitor='val_accuracy',patience=20,mode='max',restore_best_weights=True,verbose=1)
    mc = ModelCheckpoint('model.h5', monitor='val_accuracy',save_best_only=True,mode='max',verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.1, patience=4,verbose=1,mode='max')
    csvlogger = CSVLogger("./model.log")


    #plot_model(model, to_file='../../model_shape.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    hist = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,
                     validation_data=(testX, testy), callbacks=[es, mc, rlr, csvlogger])
    # matrix = plot_confusion_matrix(fit,trainX,testX)
    # plt.show(matrix)

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(ncol=2, loc='lower right')

    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(ncol=2, loc='upper right')


    try:
        os.mkdir(str(np.max(hist.history['val_accuracy'])))
    except:
        shutil.rmtree(str(np.max(hist.history['val_accuracy'])))
        os.mkdir(str(np.max(hist.history['val_accuracy'])))

    plt.tight_layout()
    plt.savefig('./'+str(np.max(hist.history['val_accuracy']))+'/loss_acc_plot.png')

    dirname = str(np.max(hist.history['val_accuracy']))
    # evaluate model
    _, accuracy1 = model.evaluate(trainX, trainy, batch_size=batch_size, verbose=1)
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
    print("train Acc: " + str(accuracy1))
    predict_x_train = model.predict(trainX)
    predict_x_test = model.predict(testX)

    #print_confusion_matrix(predict_x_train,800)
    print("")

    print_confusion_matrix(predict_x_test,200,dirname)
    print("")

    # classes_x = np.argmax(predict_x_test, axis=1)
    # classes_x2 = classes_x.reshape(150, 100)
    # print(classes_x2)

    return accuracy, np.max(hist.history['val_accuracy'])


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

def summarize_resultss(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Max accuracy: %.3f%% (+/-%.3f)' % (m, s))



# run an experiment
def run_experiment(repeats=50):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    max_accs = list()
    for r in range(repeats):
        score, max_acc = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        max_acc = max_acc * 100.0
        print('>ACC #%d: %.3f' % (r + 1, score))
        print('>MAX ACC #%d: %.3f' % (r + 1, max_acc))
        scores.append(score)
        max_accs.append(max_acc)

    # summarize results
    summarize_results(scores)
    summarize_resultss(max_accs)


# run the experiment
run_experiment()

exit()