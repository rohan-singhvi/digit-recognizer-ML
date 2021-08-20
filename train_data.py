from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

# load the test data, one is for training, the other is for testing
# inspecting into the shape of these, the x values are matrices
#  (60000 28x28 matrices to be exact)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# we will first try a multi-layer perceptron model, meaning that images must be reduced to a single vector of the pixels.
# since we know that the images are sized 28x28, by using x_train.shape[1], x_train.shape[2]
# there will be 784 pixel input values.

# reduction of the 28x28 image to a 784 vector for each of the 60k images
total_pixels = x_train.shape[1] * x_train.shape[2]  # both the training and testing data has the same sized images.
# Use of 'float32' may seem odd, however this is to reduce the memory requirements
# this is done by foring the precision of the pixel values to be 32 bit
x_train = x_train.reshape((x_train.shape[0], total_pixels)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], total_pixels)).astype('float32')

#after printing the train and test values, it was discovered that the inputs are values 0-255
# inputs must be normalized, and since we know the max is 255, we can divide all values by 255 to get it between 0-1
x_train /= 255
x_test /= 255

# we must use a one hot encoding of the class values, which will transform the 
# vector of class integers into a binary matrix
# *one hot encoding example: suppose you have a 'flower' feature which can take the values
# 'daffodil', 'lily', and 'rose'. One hot encoding would convert the 'flower'
# feature to three features:
# 'is_dafodil', 'is_lily', and 'is_rose' as 1s and 0s (in a 3x3, in this case, table)
# for an example with out data, y_train[0] before the encoding is just '5',
# however afterwards y_train becomes '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]'
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
total_classes = y_test.shape[1]

# creating a simple neural network model, defined in a function for ease of use
def simple_model():
	model = Sequential()
	model.add(Dense(total_pixels, input_dim=total_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(total_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = simple_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Base error: %.2f%%" %(100 - scores[1]*100))
