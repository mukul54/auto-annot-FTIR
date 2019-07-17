from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas
import cv2

import tensorflow as tf
from keras.models import *
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization,Activation, add, concatenate
from keras.initializers import glorot_uniform

def residual_block(X, filters, strides):

	X_shortcut = X

	X = Conv2D(filters = filters[0], kernel_size = (3,3), padding = 'same', strides = strides[0], kernel_initializer = glorot_uniform(seed = 0))(X)
	X = Activation(activation = 'relu')(X)
	X = BatchNormalization()(X)
	

	X = Conv2D(filters = filters[1], kernel_size = (3,3), padding = 'same', strides = strides[1], kernel_initializer = glorot_uniform(seed = 0))(X)
	X = Activation(activation = 'relu')(X)
	X = BatchNormalization()(X)
	
	
	X_shortcut = Conv2D(filters = filters[1], kernel_size = (1,1), strides = strides[0], kernel_initializer = glorot_uniform(seed = 0))(X_shortcut)
	X_shortcut = BatchNormalization()(X_shortcut)

	X = add([X, X_shortcut])
	X = Activation(activation = 'relu')(X)

	return X

def encoder(X):

	output = []

	X_shortcut = X

	X = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', strides = (1, 1), kernel_initializer = glorot_uniform(seed = 0))(X)
	X = Activation(activation = 'relu')(X)
	X = BatchNormalization()(X)

	X = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', strides = (1, 1), kernel_initializer = glorot_uniform(seed = 0))(X)


	X_shortcut = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), kernel_initializer = glorot_uniform(seed = 0))(X)
	X_shortcut = BatchNormalization()(X_shortcut)

	X = add([X, X_shortcut])
	#first branch of the decoder
	output.append(X)

	X = residual_block(X, [128,128], [(2, 2), (1,1)])
	#second branch of the decoder
	output.append(X)

	X = residual_block(X, [256, 256], [(2, 2), (1, 1)])
	#third branch of the decoder
	output.append(X)

	return output

def decoder(X, encoder_output):

	X = UpSampling2D(size = (2, 2))(X)
	X = concatenate([X, encoder_output[2]], axis = 3)
	X = residual_block(X, [256, 256], [(1, 1), (1, 1)])

	X = UpSampling2D(size = (2, 2))(X)
	X = concatenate([X, encoder_output[1]], axis = 3)
	X = residual_block(X, [128, 128], [(1, 1), (1, 1)])

	X = UpSampling2D(size = (2, 2))(X)
	X = concatenate([X, encoder_output[0]], axis = 3)
	X = residual_block(X, [64, 64], [(1, 1), (1, 1)])

	return X


def build_model(input_shape):

	X = Input(shape = input_shape)

	encoder_output = encoder(X)

	X1 = residual_block(encoder_output[2], [512, 512], [(2, 2), (1,1)])

	X1 = decoder(X1, encoder_output)

	X1 = Conv2D(filters = 1, kernel_size = (1, 1), activation = 'sigmoid', kernel_initializer = glorot_uniform(seed = 0))(X1)

	return Model(input = X, output = X1)
