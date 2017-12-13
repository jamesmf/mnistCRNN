# mnistCRNN
Testing TimeDistributedConvolution layers with GRU layers

## Requirements
This code is built on keras, and is a demonstration of how to use the new TimeDistributed wrapper in 1.0.2 for convolutional-recurrent neural networks (previously compatible with 0.3.2 - for backwards compatibility find previous commits or change the Reshape layer). It was previously built on [keras-extra](https://github.com/anayebi/keras-extra/), but keras has since merged TimeDistributed as a wrapper for arbitrary layers.

## Task
The addMNISTrnn.py script downloads the MNIST dataset and creates training vectors with varying numbers of images in them. It then trains a CRNN on the sequences to predict the sum of the digits.

## Validation
The model achieves a RMSE of 1.10 on the task of guessing the sum of 1 to 8 digits, with the baseline of guessing the distribution's mean having RMSE 11.81. 

## Notes
A simpler model can do just as well on this task, but this one has multiple conv layers and multiple GRU layers in order to demonstrate how they interact.

```
#define our time-distributed setup
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
```

## Example Output
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929148/4ce5c8c8-cf3f-11e5-835c-4d9eacff485f.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929147/4ce599c0-cf3f-11e5-90ea-84b06bcef147.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929149/4ce7eafe-cf3f-11e5-932a-fa9f9ea52a70.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929150/4ce8e332-cf3f-11e5-8dc2-6e17efd28588.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929153/4ceb8f92-cf3f-11e5-8da0-31b1779fd69f.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929152/4ceb73cc-cf3f-11e5-9e70-ecf16ab83ebf.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929159/4cf3ece6-cf3f-11e5-9255-6800372be51f.jpg)
![alt1](https://cloud.githubusercontent.com/assets/7809188/12929158/4cf2136c-cf3f-11e5-8bfb-6995eca11f9d.jpg)

The output for this input is 38.82
