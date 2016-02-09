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
from keras.datasets import mnist
import scipy.misc as mi
from keras.models import model_from_json

    
def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model

#define some run parameters
maxToAdd        = 8
hidden_units    = 150
examplesPer     = 10
size            = 28

model   = loadThatModel("../models/basicRNN")

# the data, shuffled and split between train and test sets
(X_train_raw, y_train_temp), (X_test_raw, y_test_temp) = mnist.load_data()

#basic image processing
X_train_raw     = X_train_raw.astype('float32')
X_test_raw      = X_test_raw.astype('float32')
X_train_raw     /= 255
X_test_raw      /= 255


#run epochs of sampling data then training
y_test      = []
X_test      = np.zeros((examplesPer,maxToAdd,1,size,size))
inds        = []
for i in range(0,examplesPer):
    output      = np.zeros((maxToAdd,1,size,size))
    numToAdd    = np.ceil(np.random.rand()*maxToAdd)
    indices     = np.random.choice(X_test_raw.shape[0],size=numToAdd)
    example     = X_test_raw[indices]
    exampleY    = y_test_temp[indices]
    output[0:numToAdd,0,:,:] = example
    X_test[i,:,:,:,:] = output
    y_test.append(np.sum(exampleY))
    inds.append(indices)

X_test  = np.array(X_test)
y_test  = np.array(y_test)       

preds   = model.predict(X_test)
#print(preds[:10])
for num, ans in enumerate(y_test):
    
    images  = np.zeros((maxToAdd,1,size,size))
    xtr     = X_test_raw[inds[num]]
    images[0:xtr.shape[0],0,:,:] = xtr
    for num2, image in enumerate(images):    
        print(image.shape)
        image   = image[0,:,:]
        mi.imsave("../images/image"+str(num2)+".jpg",image)
    print(ans, preds[num])
    for num3 in range(0,5):
        output                  = np.zeros((maxToAdd,1,size,size))
        example                 = X_test[num][0:num3+1]
        output[0:num3+1,:,:,:]  = example
        output                  = np.reshape(output,(1,output.shape[0],output.shape[1],output.shape[2],output.shape[3]))
        tempPred                = model.predict(output)
        print("with ",num3," images: ",tempPred)
    stop=raw_input("")

