""" Mixing PixelCNN and VAE in Blocks"""
import sys
import numpy as np
import theano
from theano import tensor as T

from blocks.algorithms import GradientDescent, Adam, RMSProp
from blocks.bricks import MLP, Rectifier, Logistic
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.serialization import dump, load
from blocks.filter import VariableFilter
from blocks.roles import OUTPUT

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten

from pixelblocks import create_network as create_pixel
from vae_blocks import Sampling, VAEloss

sys.setrecursionlimit(500000)

batch_size = 16
latent_dim = 2
hidden_dim = 500
img_dim = 28
nb_epoch = 200
n_channel = 1
patience = 2
path = '/data/lisa/exp/alitaiga/Generative-models/checkpoint_pixelvae'
sources = ('features',)
train = True
resume = False
seed = 2

def create_vae(x=None, batch=batch_size):
    x = T.matrix('features') if x is None else x
    x = x / 255.

    encoder = MLP(
        activations=[Rectifier(), Logistic()],
        dims=[img_dim**2, hidden_dim, 2*latent_dim],
        weights_init=IsotropicGaussian(std=0.01, mean=0),
        biases_init=Constant(0.01),
        name='encoder'
    )
    encoder.initialize()
    z_param = encoder.apply(x)
    z_mean, z_log_std = z_param[:,latent_dim:], z_param[:,:latent_dim]
    z = Sampling(theano_seed=seed).apply([z_mean, z_log_std], batch=batch_size)

    decoder = MLP(
        activations=[Rectifier(), Logistic()],
        dims=[latent_dim, hidden_dim, img_dim**2],
        weights_init=IsotropicGaussian(std=0.01, mean=0),
        biases_init=Constant(0.01),
        name='decoder'
    )
    decoder.initialize()
    x_reconstruct = decoder.apply(z)

    cost = VAEloss().apply(x, x_reconstruct, z_mean, z_log_std)
    cost.name = 'vae_cost'
    return cost

def create_network(batch=batch_size):
    x = T.matrix('features')
    vae_cost = create_vae(x,batch=batch)
    pixel_cost = create_pixel(x,batch=batch)
    total_cost = vae_cost + pixel_cost
    total_cost.name = 'total_cost'
    return total_cost

def prepare_opti(cost, test):
    model = Model(cost)

    algorithm = GradientDescent(
        cost=cost,
        parameters=model.parameters,
        step_rule=RMSProp(),
        on_unused_sources='ignore'
    )

    extensions = [
        FinishAfter(after_n_epochs=nb_epoch),
        FinishIfNoImprovementAfter(notification_name='test_cross_entropy', epochs=patience),
        TrainingDataMonitoring(
            [algorithm.cost],
            prefix="train",
            after_epoch=True),
        DataStreamMonitoring(
            [algorithm.cost],
            test_stream,
            prefix="test"),
        Printing(),
        ProgressBar(),
        #Checkpoint(path, after_epoch=True)
    ]

    if resume:
        print "Restoring from previous breakpoint"
        extensions.extend([
            Load(path)
        ])
    return model, algorithm, extensions


if __name__ == '__main__':
    mnist = MNIST(("train",), sources=sources)
    mnist_test = MNIST(("test",), sources=sources)
    training_stream = Flatten(
        DataStream(
            mnist,
            iteration_scheme=ShuffledScheme(mnist.num_examples, batch_size)
        ),
        which_sources=sources
    )
    # import ipdb; ipdb.set_trace()
    test_stream = Flatten(
        DataStream(
            mnist_test,
            iteration_scheme=ShuffledScheme(mnist_test.num_examples, batch_size)
        ),
        which_sources=sources
    )
    "Print data loaded"

    if train:
        cost = create_network()
        model, algorithm, extensions = prepare_opti(cost, test_stream)

        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=training_stream,
            model=model,
            extensions=extensions
        )
        main_loop.run()
        dump(main_loop.model, open('pixelvae.pkl', 'w'))
    else:
        model = load(open('pixelvae.pkl', 'r'))
