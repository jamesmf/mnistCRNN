# mnistCRNN
This is a example of how to use TimeDistributed() to wrap the complete model(take MobileNet for example).

## Requirements
This code is built on keras, using tensorflow by default.

## Task
The addMNIST_model.py script downloads the MNIST dataset, the MobileNet model weight and creates training vectors with varying numbers of images in them. It then trains a CRNN on the sequences to predict the sum of the digits.

## Validation
The model achieves a RMSE of 0.65 on the task of guessing the sum of 1 to 8 digits, with the baseline of guessing the distribution's mean having RMSE 11.90. 

## Notes
A simpler model can do just as well on this task, but this one has multiple conv layers and multiple GRU layers in order to demonstrate how they interact.

Because of the commutative nature of the task, a purely convolutional approach would be significantly faster given the fixed `numToAdd` size. Shared-parameters `MobileNet` stacks could process all the digits at once. But that's not the point of the demo!

```
#define our time-distributed setup
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
```
