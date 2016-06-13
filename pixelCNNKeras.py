"""Impleting PixelCNN in keras"""

from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D
from keras.layers.core import Activation
from keras import backend as K
from keras.objectives import binary_crossentropy

import matplotlib.pyplot as plt
import numpy as np

batch_size = 100
mnist_dim = 784
latent_dim = 2
hidden_dim = 500
epsilon = 0.01
nb_epoch = 40
first_layer = (32, 7, 7)
second_layer = (32, 3, 3)
third_layer = (64, 3, 3)
fourth_layer = (128, 1, 1)
action = 'sigmoid'

# PixelCNN architecture, no pooling layer
x = Input(batch_shape=(batch_size,mnist_dim))

x1 = Convolution2D(*first_layer)(x)

x2 = Convolution2D(*second_layer)(x1)
x3 = Convolution2D(*second_layer)(x2)
x4 = Convolution2D(*second_layer)(x3)
x5 = Convolution2D(*second_layer)(x4)
x6 = Convolution2D(*second_layer)(x5)
x7 = Convolution2D(*third_layer, activation='relu')(x6)

x8 = Convolution2D(*fourth_layer, activation='relu')(x7)
x9 = Convolution2D(*fourth_layer)(x8)
y = Activation(activation)
