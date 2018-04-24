'''
Basic demonstration of the capabilities of the CRNN using TimeDistributed
wrapper. The MobileNet model is encapsulated by the TimeDistributed wrapper.
Processes an MNIST image (or blank square) at each time step and sums the
digits. Learning is based on the sum of the digits, not explicit labels
on each digit.
Considering the minimum input of MobileNet, the mnist images are resize to
128*128 and the mode is changed to 'RGB'.
'''

from __future__ import print_function
import numpy as np
import gc
import keras
import keras.backend as K
from PIL import Image
from keras.datasets import mnist
from keras.models import Model
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input
from keras.preprocessing.image import array_to_img
from keras.applications.mobilenet import relu6, DepthwiseConv2D


custom_objects = {
    'relu6'          : relu6,
    'DepthwiseConv2D': DepthwiseConv2D
}

# for reproducibility
np.random.seed(2016)
K.set_image_dim_ordering('tf')

# define some run parameters
batch_size = 32
nb_epochs = 15
examplesPer = 60000
maxToAdd = 8
hidden_units = 200
size = 128

# the data, shuffled and split between train and test sets
(X_train_raw, y_train_temp), (X_test_raw, y_test_temp) = mnist.load_data()

print('X_train_raw shape:', X_train_raw.shape)
print('X_test_raw shape:', X_test_raw.shape)
print(X_train_raw.shape[0], 'train samples')
print(X_test_raw.shape[0], 'test samples')
print("Building model")

# define our time-distributed setup
inp = Input(shape=(maxToAdd, size, size, 3))
base_model = keras.applications.MobileNet(input_shape=(size, size, 3),
                                 include_top=False,
                                 weights='imagenet',
                                 input_tensor=None,
                                 pooling='avg',
                                 classes=10)
x = TimeDistributed(base_model)(inp)
x = GRU(units=100, return_sequences=True)(x)
x = GRU(units=50, return_sequences=False)(x)
x = Dropout(.2)(x)
x = Dense(1)(x)
model = Model(inp, x)

rmsprop = RMSprop()
model.compile(loss='mean_squared_error', optimizer=rmsprop)

# run epochs of sampling data then training
for ep in range(0, nb_epochs):
    X_train = np.zeros((examplesPer, maxToAdd, size, size, 3), dtype='float16')
    y_train = []

    for i in range(0, examplesPer):
        # initialize a training example of max_num_time_steps, im_size, im_size, 3
        output = np.zeros((maxToAdd, size, size, 3), dtype='float16')
        # decide how many MNIST images to put in that tensor
        numToAdd = int(np.ceil(np.random.rand()*maxToAdd))
        # sample that many images
        indices = np.random.choice(X_train_raw.shape[0], size=numToAdd)
        example = X_train_raw[indices]
        example_ = []
        for img in example:
            img = np.expand_dims(img, axis=-1)
            im = array_to_img(img).convert('RGB').resize((size, size), resample=Image.BILINEAR)
            example_.append(np.asarray(im) / 255.0)
        exampleY = y_train_temp[indices]
        output[0:numToAdd, :, :, :] = np.array(example_, dtype='float16')
        X_train[i, :, :, :, :] = output
        y_train.append(np.sum(exampleY))

    y_train = np.array(y_train)

    if ep == 0:
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=1,
              verbose=1)

    # save the model
jsonstring  = model.to_json()
with open("../models/basicRNN.json", 'wb') as f:
   f.write(jsonstring)
model.save_weights("../models/basicRNN.h5", overwrite=True)

# del X_train_raw, y_train_temp, X_train, y_train
# gc.collect()

# Test the model
X_test = np.zeros((examplesPer, maxToAdd, size, size, 3), dtype='float16')
y_test = []
for i in range(0, examplesPer):
    output = np.zeros((maxToAdd, size, size, 3))
    numToAdd = int(np.ceil(np.random.rand()*maxToAdd))
    indices = np.random.choice(X_test_raw.shape[0], size=numToAdd)
    example = X_test_raw[indices]
    example_ = []
    for img in example:
        img = np.expand_dims(img, axis=-1)
        im = array_to_img(img).convert('RGB').resize((size, size), resample=Image.BILINEAR)
        example_.append(np.asarray(im) / 255.0)
    exampleY = y_test_temp[indices]
    output[0:numToAdd, :, :, :] = np.array(example_, dtype='float16')
    X_test[i, :, :, :, :] = output
    y_test.append(np.sum(exampleY))

X_test = np.array(X_test)
y_test = np.array(y_test)       

preds = model.predict(X_test)

# print the results of the test
print(np.sum(np.sqrt(np.mean([ (y_test[i] - preds[i][0])**2 for i in range(0,len(preds)) ]))))
print("naive guess", np.sum(np.sqrt(np.mean([ (y_test[i] - np.mean(y_test))**2 for i in range(0,len(y_test)) ]))))
