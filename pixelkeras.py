"""Impleting PixelCNN in keras"""
from __future__ import division

from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Convolution2D, merge
from keras.layers.core import Activation, Reshape, Permute
from keras import backend as K

import theano
from theano import tensor as T

import numpy as np

batch_size = 16
mnist_dim = 28
nb_epoch = 40
n_channel = 1
patience = 3

MODE = '256ary'  # choice with 'binary' and '256ary

if MODE == 'binary':
    activation = 'sigmoid'
    loss = 'binary_crossentropy'
elif MODE == '256ary':
    activation = 'softmax'
    loss = 'categorical_crossentropy'
n_layer = 6
res_connections = True
first_layer = (32, 7, 7)
second_layer = (32, 3, 3)
third_layer = (256 if MODE == '256ary' else 1, 1, 1)


class Convolution2DNoFlip(Convolution2D):
    def __init__(self, *args, **kwargs):
        self.mask = kwargs.pop('mask', None)
        super(Convolution2DNoFlip, self).__init__(*args, **kwargs)


    def call(self, x, mask=None, strides=(1, 1)):
        if self.mask:
            mask = np.ones(self.W_shape, dtype=theano.config.floatX)
            middle = self.W_shape[2] // 2
            mask[middle+1:,:] = 0.
            if self.mask == 'A':
                mask[middle,middle:] = 0.
            elif self.mask == 'B':
                mask[middle,middle+1:] = 0.
            self.W = self.W * mask

        if self.border_mode == 'same':
            th_border_mode = 'half'
            np_kernel = self.W.eval()
        elif self.border_mode == 'valid':
            th_border_mode = 'valid'
        output = T.nnet.conv2d(x, self.W,
                             border_mode=th_border_mode,
                             subsample=self.subsample,
                             filter_shape=self.W_shape,
                             filter_flip=False)

        if self.border_mode == 'same':
            if np_kernel.shape[2] % 2 == 0:
                output = output[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
            if np_kernel.shape[3] % 2 == 0:
                output = output[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        return output


def create_network():
    # PixelCNN architecture, no pooling layer
    x = Input(batch_shape=(batch_size,n_channel,mnist_dim,mnist_dim))

    # First layer using  mask A
    x_ = Convolution2DNoFlip(*first_layer, input_shape=(1, 28, 28), border_mode='same', mask='A')(x)

    # Second type of layers using mask B
    for i in range(n_layer // 2):
        x_1 = Convolution2DNoFlip(*second_layer, activation='relu', border_mode='same', mask='B')(x_)
        x_2 = Convolution2DNoFlip(*second_layer, activation='relu', border_mode='same', mask='B')(x_1)

        if res_connections:
            x_ = merge([x_, x_2], mode='sum')
        else:
            x_ = x_2

    # 2 layers of Relu followed by 1x1 conv
    x_ = Convolution2DNoFlip(64, 1, 1, activation='relu', border_mode='same', mask='B')(x_)
    x_ = Convolution2DNoFlip(128, 1, 1, activation='relu', border_mode='same', mask='B')(x_)

    # Depending on the output
    x_ = Convolution2DNoFlip(*third_layer,border_mode='same', mask='B')(x_)

    if MODE == '256ary':
        x_ = Reshape((256, mnist_dim**2))(x_)
        x_ = Permute((2,1))(x_)

    y = Activation(activation)(x_)

    model = Model(x, y)
    model.compile(optimizer='adagrad', loss=cost)
    print "Model compiled"
    return model

def cost(pred, truth):
    pred = pred.reshape((batch_size*784,256))
    truth = T.cast(truth[:,:,0], 'int64').flatten()
    #truth = T.cast(truth.flatten(), 'int64')
    return T.nnet.categorical_crossentropy(pred, truth).mean()

def generate_samples(model):
    pass

if __name__ == '__main__':

    PixelCNN = create_network()
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape((len(x_train), n_channel, mnist_dim, mnist_dim))
    x_test = x_test.reshape((len(x_test), n_channel, mnist_dim, mnist_dim))

    print "Starting training"
    PixelCNN.fit(
        x_train,
        x_train,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[EarlyStopping(patience=patience)]
    )
    PixelCNN.save_weights('pixelcnn_weights.h5')
