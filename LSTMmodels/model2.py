
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import random
import keras
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
import matplotlib.pyplot as plt
#from keras.initializations import norRemal, identity
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, Adadelta
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import json
img_shape= (10,64,64,3)
model = Sequential()

model.add(Lambda(lambda x: x * 1./127.5 - 1,input_shape=(img_shape), output_shape=(img_shape), name='Normalization'))
model.add(TimeDistributed(Convolution2D(8, 4, 4, border_mode='valid'), input_shape=(10,64,64,3)))
        #model.add(Activation('relu'))
model.add(ELU())
model.add(TimeDistributed(Convolution2D(16, 3, 3, border_mode='valid')))
model.add(ELU())
        #model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
        #model.add(Activation('relu'))
model.add(TimeDistributed(Convolution2D(8, 3, 3, border_mode='valid')))
model.add(ELU())
        #model.add(Activation('relu'))
        #model.add(Reshape((maxToAdd,np.prod(model.output_shape[-3:])))) #this line updated to work with keras 1.0.2
model.add(TimeDistributed(Flatten()))
model.add(Activation('relu'))
model.add(LSTM(output_dim=100,return_sequences=True))
model.add(LSTM(output_dim=50,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation("linear")) 
adam = Adam(lr = 0.0001)
        #rmsprop = RMSprop()
model.compile(loss='mean_squared_error', optimizer=adam,metrics=['accuracy'])
#model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])

model.summary()

with open('modelNew.json', 'w') as f:
    json.dump(model.to_json(), f)
