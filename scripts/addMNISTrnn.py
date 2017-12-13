'''
Basic demonstration of the capabilities of the CRNN using TimeDistributed
wrapper. Processes an MNIST image (or blank square) at each time step and
sums the digits. Learning is based on the sum of the digits, not explicit
labels on each digit.
'''

from __future__ import print_function
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input


# for reproducibility
np.random.seed(2016)
K.set_image_dim_ordering('tf')

# define some run parameters
batch_size = 32
nb_epochs = 1
examplesPer = 60000
maxToAdd = 8
hidden_units = 200
size = 28

# the data, shuffled and split between train and test sets
(X_train_raw, y_train_temp), (X_test_raw, y_test_temp) = mnist.load_data()

# basic image processing
X_train_raw = X_train_raw.astype('float32')
X_test_raw = X_test_raw.astype('float32')
X_train_raw /= 255
X_test_raw /= 255


print('X_train_raw shape:', X_train_raw.shape)
print(X_train_raw.shape[0], 'train samples')
print(X_test_raw.shape[0], 'test samples')
print("Building model")

# define our time-distributed setup
inp = Input(shape=(maxToAdd, size, size, 1))
x = TimeDistributed(Conv2D(8, (4, 4), padding='valid', activation='relu'))(inp)
x = TimeDistributed(Conv2D(16, (4, 4), padding='valid', activation='relu'))(x)
x = TimeDistributed(Flatten())(x)
x = GRU(units=100, return_sequences=True)(x)
x = GRU(units=50, return_sequences=False)(x)
x = Dropout(.2)(x)
x = Dense(1)(x)
model = Model(inp, x)

rmsprop = RMSprop()
model.compile(loss='mean_squared_error', optimizer=rmsprop)

# run epochs of sampling data then training
for ep in range(0, nb_epochs):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    X_train = np.zeros((examplesPer, maxToAdd, size, size, 1))

    for i in range(0, examplesPer):
        # initialize a training example of max_num_time_steps,im_size,im_size
        output = np.zeros((maxToAdd, size, size, 1))
        # decide how many MNIST images to put in that tensor
        numToAdd = int(np.ceil(np.random.rand()*maxToAdd))
        # sample that many images
        indices = np.random.choice(X_train_raw.shape[0], size=numToAdd)
        example = X_train_raw[indices]
        # sum up the outputs for new output
        exampleY = y_train_temp[indices]
        output[0:numToAdd, :, :, 0] = example
        X_train[i, :, :, :, :] = output
        y_train.append(np.sum(exampleY))

    y_train = np.array(y_train)

    if ep == 0:
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=1,
              verbose=1)

# Test the model
X_test = np.zeros((examplesPer, maxToAdd, size, size, 1))
for i in range(0, examplesPer):
    output = np.zeros((maxToAdd, size, size, 1))
    numToAdd = int(np.ceil(np.random.rand()*maxToAdd))
    indices = np.random.choice(X_test_raw.shape[0], size=numToAdd)
    example = X_test_raw[indices]
    exampleY = y_test_temp[indices]
    output[0:numToAdd, :, :, 0] = example
    X_test[i, :, :, :, :] = output
    y_test.append(np.sum(exampleY))

X_test = np.array(X_test)
y_test = np.array(y_test)       

preds = model.predict(X_test)

# print the results of the test
print(np.sum(np.sqrt(np.mean([ (y_test[i] - preds[i][0])**2 for i in range(0,len(preds)) ]))))
print("naive guess", np.sum(np.sqrt(np.mean([ (y_test[i] - np.mean(y_test))**2 for i in range(0,len(y_test)) ]))))

# save the model
#jsonstring  = model.to_json()
#with open("../models/basicRNN.json",'wb') as f:
#    f.write(jsonstring)
#model.save_weights("../models/basicRNN.h5",overwrite=True)

