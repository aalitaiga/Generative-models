""" Implementing VAE in blcks"""

from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import application, MLP, Rectifier, Random, Logistic, Tanh
from blocks.bricks.cost import Cost
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.initialization import IsotropicGaussian, Constant
from blocks.serialization import dump, load

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten

import theano
from theano import tensor as T
import matplotlib.pyplot as plt
import numpy as np

batch_size = 100
mnist_dim = 784
latent_dim = 2
hidden_dim = 500
epsilon = 0.01
nb_epoch = 50
patience = 1
seed = 2
sources = (u'features',)
train = True

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

class VAEloss(Cost):

    @application
    def apply(self, x_, x_r, z_m, z_lstd):
        rec_loss = T.nnet.binary_crossentropy(x_r, x_).mean(axis=-1).mean()
        kl_loss = - 0.5 * T.sum(1 + 2*z_lstd - z_m**2 - T.exp(2*z_lstd), axis=1).mean()
        return rec_loss + kl_loss

def create_network():
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

    encoder.initialize()
    decoder.initialize()

    cost = VAEloss().apply(x, x_reconstruct, z_mean, z_log_std)
    cost.name = 'total_cost'
    return cost, encoder, decoder


def prepare_opti(cost, test):
    model = Model(cost)
    algorithm = GradientDescent(cost=cost, parameters=model.parameters,
        step_rule=Adam(), on_unused_sources='ignore')

    extensions = [
        FinishAfter(after_n_epochs=nb_epoch),
        FinishIfNoImprovementAfter(notification_name='test_total_cost', epochs=patience),
        TrainingDataMonitoring(
            [algorithm.cost],
            after_epoch=True),
        DataStreamMonitoring(
            [algorithm.cost],
            test,
            prefix="test"),
        Printing(),
        ProgressBar()
    ]
    return model, algorithm, extensions

if __name__ == '__main__':
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

    if train:
        cost, encoder, decoder = create_network()
        model, algorithm, extensions = prepare_opti(cost, data_stream_test)

        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=data_stream,
            model=model,
            extensions=extensions
        )

        main_loop.run()
        dump(main_loop.model, open('weights.pkl', 'w'))
        model = main_loop.model
    else:
        model = load(open('weights.pkl', 'r'))
    for brick in model.top_bricks :
        if brick.name == 'encoder' :
            encoder = brick
        if brick.name == 'decoder' :
            generator = brick

    print 'Training finished model saved'


    x_test, y_test = MNIST(("test",)).get_data(state=None, request=range(10000))
    a = T.matrix(u'features')
    b = encoder.apply(a)
    c = generator.apply(a)
    encode = theano.function([a], [b])
    generate = theano.function([a], [c])

    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encode(x_test.reshape((10000,784)))[0]
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test.reshape((10000,)))
    plt.colorbar()
    plt.show()

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
            x_decoded = generate(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()
