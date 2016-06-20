""" Implementing VAE in blcks"""

from blocks.algorithms import GradientDescent, Adam, Scale
from blocks.bricks import application, Linear, MLP, Rectifier, Random, Logistic, Tanh
from blocks.bricks.cost import Cost
from blocks.initialization import IsotropicGaussian, Constant
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten

from theano import tensor as T
import matplotlib.pyplot as plt
import numpy as np

batch_size = 100
mnist_dim = 784
latent_dim = 2
hidden_dim = 500
epsilon = 1
nb_epoch = 40
patience = 2
seed = 3
sources = (u'features',)

# Encoder
x = T.matrix(u'features') / 255.
encoder = MLP(
    activations=[Rectifier(), Logistic()],
    dims=[mnist_dim, hidden_dim, 2*latent_dim],
    weights_init=IsotropicGaussian(std=0.01, mean=0),
    biases_init=Constant(0.01),
    name='encoder'
)
z_param = encoder.apply(x)
z_mean, z_log_std = z_param[:,latent_dim:], z_param[:,:latent_dim]

class Sampling(Random):

    @application
    def apply(self, args):
        mean, log_std = args
        eps = self.theano_rng.normal(
            size=(batch_size,latent_dim),
            avg=0.0,
            std=epsilon
        )
        return mean + T.exp(log_std) * eps

z = Sampling(theano_seed=seed).apply([z_mean, z_log_std])

# Decoder
decoder = MLP(
    activations=[Rectifier(), Logistic()],
    dims=[latent_dim, hidden_dim, mnist_dim],
    weights_init=IsotropicGaussian(std=0.01, mean=0),
    biases_init=Constant(0.01),
    name='decoder'
)

x_reconstruct = decoder.apply(z)

class VAEloss(Cost):

    @application
    def apply(self, x_, x_r):
        rec_loss = T.nnet.binary_crossentropy(x_r, x_).mean(axis=-1).mean()
        kl_loss = - 0.5 * T.sum(1 + 2*z_log_std - z_mean**2 - T.exp(2*z_log_std), axis=1).mean()
        return rec_loss + kl_loss

encoder.initialize()
decoder.initialize()

cost = VAEloss().apply(x, x_reconstruct)
cost.name = 'total_cost'
model = Model(cost)

#import ipdb; ipdb.set_trace()
algorithm = GradientDescent(cost=cost, parameters=model.parameters,
    step_rule=Scale(0.), on_unused_sources='ignore')

mnist = MNIST(("train",))
data_stream = Flatten(
    DataStream(
        mnist,
        iteration_scheme=ShuffledScheme(mnist.num_examples, batch_size)
    ),
    which_sources=sources
)

mnist_test = MNIST(("test",))
data_stream_test = Flatten(
    DataStream(
        mnist_test,
        iteration_scheme=ShuffledScheme(mnist_test.num_examples, batch_size)
    ),
    which_sources=sources
)

extensions = [
    FinishAfter(after_n_epochs=nb_epoch),
    FinishIfNoImprovementAfter(notification_name='test_cross_entropy', epochs=patience),
    TrainingDataMonitoring(
        [algorithm.cost],
        after_epoch=True),
    DataStreamMonitoring(
        [algorithm.cost],
        data_stream_test,
        prefix="test"),
    Printing(),
    ProgressBar()
]

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=data_stream,
    model=model,
    extensions=extensions
)
main_loop.run()

# print 'Training finished model saved'
#
# # build a model to project inputs on the latent space
# encoder = Model(x, z_mean)
#
# # display a 2D plot of the digit classes in the latent space
# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# plt.show()
#
# # build a digit generator that can sample from the learned distribution
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h(decoder_input)
# _x_decoded_mean = decoder_mean(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)
#
# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # we will sample n points within [-15, 15] standard deviations
# grid_x = np.linspace(-15, 15, n)
# grid_y = np.linspace(-15, 15, n)
#
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]]) * epsilon
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit
#
# plt.figure(figsize=(10, 10))
# plt.imshow(figure)
# plt.show()
