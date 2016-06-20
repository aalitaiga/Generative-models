"""Impleting VAE in keras"""

from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Lambda
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

# Encoder
x = Input(batch_shape=(batch_size,mnist_dim))
h = Dense(hidden_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_std = Dense(latent_dim)(h)

def sampling(args):
    mean, std = args
    eps = K.random_normal(
        shape=(batch_size,latent_dim),
        mean=0.0,
        std=epsilon
    )
    return mean + std * eps

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_std])

# Decoder
decoder_h = Dense(hidden_dim, activation='relu')
decoder_mean = Dense(mnist_dim, activation='sigmoid')
h_decod = decoder_h(z)
x_reconstruct = decoder_mean(h_decod)

def vae_loss(x_, x_reconstruct):
    rec_loss = binary_crossentropy(x_, x_reconstruct)
    kl_loss = - 0.5 * K.mean(1 + 2*K.log(z_std + 1e-10) - z_mean**2 - z_std**2, axis=-1)
    return rec_loss + kl_loss

vae = Model(x, x_reconstruct)
vae.compile(optimizer='adam', loss=vae_loss)
print "Model compiled"

# Copy-pasted from https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print 'Starting training'
vae.fit(
    x_train,
    x_train,
    shuffle=True,
    nb_epoch=nb_epoch,
    batch_size=batch_size,
    validation_data=(x_test, x_test),
    callbacks=[EarlyStopping(patience=1)]
)

vae.save_weights('vae_weights.h5')
print 'Training finished model saved'

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
