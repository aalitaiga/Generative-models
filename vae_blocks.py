""" Implementing VAE in blcks"""

from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import application, Linear, MLP, Rectifier, Random, Logistic
from blocks.bricks.cost import Cost
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.monitoring import DataStreamMonitoring
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
epsilon = 0.01
nb_epoch = 40
patience = 2
seed = 2
sources = (u'features')

# Encoder
x = T.matrix(u'features')
input_to_hidden = Linear(
    name='input_to_hidden',
    input_dim=mnist_dim,
    output_dim=hidden_dim,
    weights_init=IsotropicGaussian(std=0.01, mean=0),
    biases_init=Constant(0.01)
)
input_to_hidden.initialize()
h = Rectifier().apply(input_to_hidden.apply(x))
h_to_z_mean = Linear(
    name='hidden_to_mean',
    input_dim=hidden_dim,
    output_dim=latent_dim,
    weights_init=IsotropicGaussian(std=0.01, mean=0),
    biases_init=Constant(0.01),
)
h_to_z_mean.initialize()
z_mean = h_to_z_mean.apply(h)
h_to_z_std = Linear(
    name='hidden_to_std',
    input_dim=hidden_dim,
    output_dim=latent_dim,
    weights_init=IsotropicGaussian(std=0.01, mean=0),
    biases_init=Constant(0.01),
)
h_to_z_std.initialize()
z_std = h_to_z_std.apply(h)


class Sampling(Random):

    @application
    def apply(self, args):
        mean, std = args
        eps = self.theano_rng.normal(
            size=(batch_size,latent_dim),
            avg=0.0,
            std=epsilon
        )
        return mean + std * eps

z = Sampling(theano_seed=seed).apply([z_mean, z_std])

# Decoder
decoder = MLP(
    activations=[Rectifier(), Logistic()],
    dims=[latent_dim, hidden_dim, mnist_dim],
    weights_init=IsotropicGaussian(std=0.01, mean=0),
    biases_init=Constant(0)
)
decoder.initialize()
x_reconstruct = decoder.apply(z)

class VAEloss(Cost):

    @application
    def apply(self, x_, x_r):
        rec_loss = T.nnet.binary_crossentropy(x_, x_r).sum(axis=1).mean()
        kl_loss = - 0.5 * T.mean(1 + T.log(T.square(z_std) + 1e-10) - T.square(z_mean) - T.square(z_std), axis=-1).mean()
        return rec_loss + kl_loss

cost = VAEloss().apply(x, x_reconstruct)
cg = ComputationGraph(cost)

model = Model(cost)

algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
    step_rule=Adam(), on_unused_sources='ignore')

mnist = MNIST(("train",))
data_stream = Flatten(DataStream(
    mnist,
    iteration_scheme=ShuffledScheme(mnist.num_examples, batch_size=batch_size)),
    which_sources=sources
)

mnist_test = MNIST(("test",))
data_stream_test = Flatten(DataStream(
    mnist_test,
    iteration_scheme=ShuffledScheme(mnist_test.num_examples, batch_size=batch_size)),
    which_sources=sources
)

monitor = DataStreamMonitoring(
    variables=[cost], data_stream=data_stream_test, prefix="test"
)

main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm, model=model,
    extensions=[
        monitor,
        FinishAfter(after_n_epochs=nb_epoch),
        FinishIfNoImprovementAfter(notification_name='test_cross_entropy',epochs=patience),
        ProgressBar(),
        Printing()])
main_loop.run()
