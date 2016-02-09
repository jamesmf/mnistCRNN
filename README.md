# mnistCRNN
Testing TimeDistributedConvolution layers with GRU layers

##Requirements
This code is built on keras, and is a demonstration of how to use [keras-extra](https://github.com/anayebi/keras-extra/) for convolutional-recurrent neural networks. It leverages the TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, and TimeDistributedFlatten layers.

##Task
The addMNISTrnn.py script downloads the MNIST dataset and creates training vectors with varying numbers of images in them. It then trains a CRNN on the sequences to predict the sum of the digits.

##Validation
The model achieves a RMSE of 1.17 on the task of guessing the sum of 1 to 8 digits, with the baseline of guessing the distribution's mean having RMSE 11.85. 

##Notes
A simpler model can do just as well on this task, but this one has multiple conv layers and multiple GRU layers in order to demonstrate how they interact.

Also if you want to use from_json_model to load the model, you need to add `from ..layers.extra import *` to the file `keras/utils/layer_utils.py` and `from .layers import extra` to the file `keras/models.py`

```
#define our time-distributed setup
model = Sequential()
model.add(TimeDistributedConvolution2D(8, 4, 4, border_mode='valid', input_shape=(maxToAdd,1,size,size)))
model.add(TimeDistributedMaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Activation('relu'))
model.add(TimeDistributedConvolution2D(8, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(TimeDistributedFlatten())
model.add(Activation('relu'))
model.add(GRU(output_dim=100,return_sequences=True))
model.add(GRU(output_dim=50,return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(1))

rmsprop = RMSprop()
model.compile(loss='mean_squared_error', optimizer=rmsprop)
```

##Example Output
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929148/4ce5c8c8-cf3f-11e5-835c-4d9eacff485f.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929147/4ce599c0-cf3f-11e5-90ea-84b06bcef147.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929149/4ce7eafe-cf3f-11e5-932a-fa9f9ea52a70.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929150/4ce8e332-cf3f-11e5-8dc2-6e17efd28588.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929153/4ceb8f92-cf3f-11e5-8da0-31b1779fd69f.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929152/4ceb73cc-cf3f-11e5-9e70-ecf16ab83ebf.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929159/4cf3ece6-cf3f-11e5-9255-6800372be51f.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929158/4cf2136c-cf3f-11e5-8bfb-6995eca11f9d.jpg)

The output for this input is 38.82
