'''This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

arXiv:1504.00941v2 [cs.NE] 7 Apr 201
http://arxiv.org/pdf/1504.00941v2.pdf

Optimizer is replaced with RMSprop which yields more stable and steady
improvement.

Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, TimeDistributedDense,Dropout
from keras.layers.recurrent import  GRU
from keras.layers.extra import TimeDistributedFlatten, TimeDistributedConvolution2D, TimeDistributedMaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils


def im2Window(image,wSize,stride):
    xdim    = image.shape[1] - wSize + 1
    ydim    = image.shape[0] - wSize + 1
    canvas  = np.zeros((image.shape[0]+stride-1,image.shape[1]+stride-1))
    canvas[0:image.shape[0],0:image.shape[1]] = image
    xran    = [int(np.round(val)) for val in np.linspace(0,xdim,xdim*1./stride)]
    yran    = [int(np.round(val)) for val in np.linspace(0,ydim,ydim*1./stride)]
    output  = []
    for y in yran:
        for x in xran:
            output.append(np.reshape(canvas[y:y+wSize,x:x+wSize],(1,wSize,wSize)))
    out     = np.array(output)
    return out

batch_size      = 32
stride          = 3
nb_classes      = 10
nb_epochs       = 200
hidden_units    = 100
wSize           = 15

learning_rate   = 1e-6
clip_norm       = 1.0

# the data, shuffled and split between train and test sets
(X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()

#cutoff          = 10000
#X_train_raw     = X_train_raw[:cutoff]
#X_test_raw      = X_test_raw[:cutoff]
#y_train         = y_train[:cutoff]
#y_test          = y_test[:cutoff]

#scale and reformat the data, print its shape
X_train_raw = X_train_raw.astype('float32')
X_test_raw = X_test_raw.astype('float32')
X_train_raw /= 255
X_test_raw /= 255
print('X_train_raw shape:', X_train_raw.shape)
print(X_train_raw.shape[0], 'train samples')
print(X_test_raw.shape[0], 'test samples')


#convert from 
X_train  = []
X_test   = []
[X_train.append(im2Window(image,wSize,stride)) for image in X_train_raw]
[X_test.append(im2Window(image,wSize,stride)) for image in X_test_raw]
X_train     = np.array(X_train)
X_test      = np.array(X_test)


Y_train     = np_utils.to_categorical(y_train, nb_classes)
Y_test      = np_utils.to_categorical(y_test, nb_classes)
# convert class vectors to binary class matrices
#Y_train = np.reshape(np_utils.to_categorical(y_train, nb_classes),(y_train.shape[0],1,nb_classes))
#Y_test  = np.reshape(np_utils.to_categorical(y_test, nb_classes),(y_test.shape[0],1,nb_classes))



n_timesteps     = len(X_train[0])
print(n_timesteps)
print(X_train.shape)
print(Y_train.shape)
#stop=raw_input("")

print('Building model...')
model = Sequential()
model.add(TimeDistributedConvolution2D(8, 4, 4, border_mode='valid', input_shape=(n_timesteps,1, X_train.shape[3],X_train.shape[4])))
model.add(TimeDistributedMaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Activation('relu'))
model.add(TimeDistributedFlatten())
model.add(Activation('relu'))
model.add(GRU(output_dim=100,return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])





#
#print('Compare to LSTM...')
#model = Sequential()
#model.add(LSTM(hidden_units, input_shape=X_train.shape[1:]))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
#rmsprop = RMSprop(lr=learning_rate)
#model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
#
#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
#          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
#
#scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
#print('LSTM test score:', scores[0])
#print('LSTM test accuracy:', scores[1])
