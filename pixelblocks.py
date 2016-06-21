""" Implementing pixelCNN in Blocks"""
import sys
import numpy as np
import theano
from theano import tensor as T

from blocks.algorithms import GradientDescent, Adam, RMSProp
from blocks.bricks.conv import ConvolutionalSequence, Convolutional
from blocks.bricks import application, Rectifier, Softmax, Random
from blocks.bricks.cost import CategoricalCrossEntropy
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

sys.setrecursionlimit(500000)

batch_size = 16
img_dim = 28
nb_epoch = 200
n_channel = 1
patience = 2
path = '/data/lisa/exp/alitaiga/Generative-models/checkpoint'
sources = ('features',)
train = True
resume = False
seed = 2

MODE = '256ary'  # choice with 'binary' and '256ary

if MODE == 'binary':
    activation = 'sigmoid'
    loss = 'binary_crossentropy'
elif MODE == '256ary':
    activation = 'softmax'
    loss = 'categorical_crossentropy'
n_layer = 6
res_connections = True
first_layer = ((7, 7), 32, n_channel)
second_layer = ((3, 3), 32, 32)
third_layer = (256 if MODE == '256ary' else 1, 1, 1)

class ConvolutionalNoFlip(Convolutional) :
    def __init__(self, *args, **kwargs):
        self.mask = kwargs.pop('mask', None)
        super(ConvolutionalNoFlip, self).__init__(*args, **kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Perform the convolution.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            A 4D tensor with the axes representing batch size, number of
            channels, image height, and image width.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A 4D tensor of filtered images (feature maps) with dimensions
            representing batch size, number of filters, feature map height,
            and feature map width.

            The height and width of the feature map depend on the border
            mode. For 'valid' it is ``image_size - filter_size + 1`` while
            for 'full' it is ``image_size + filter_size - 1``.

        """
        if self.image_size == (None, None):
            input_shape = None
        else:
            input_shape = (self.batch_size, self.num_channels)
            input_shape += self.image_size

        if self.mask:
            filter_shape = (self.num_filters, self.num_channels) + self.filter_size
            mask = np.ones(filter_shape, dtype=theano.config.floatX)
            middle = filter_shape[2] // 2
            mask[middle+1:,:] = 0.
            if self.mask == 'A':
                mask[middle,middle:] = 0.
            elif self.mask == 'B':
                mask[middle,middle+1:] = 0.
            self.W.set_value(self.W.get_value() * mask)
            assert self.W.get_value().shape == filter_shape

        output = self.conv2d_impl(
            input_, self.W,
            input_shape=input_shape,
            subsample=self.step,
            border_mode=self.border_mode,
            filter_shape=((self.num_filters, self.num_channels) +
                           self.filter_size),
            filter_flip=False)
        if getattr(self, 'use_bias', True):
            if self.tied_biases:
                output += self.b.dimshuffle('x', 0, 'x', 'x')
            else:
                output += self.b.dimshuffle('x', 0, 1, 2)
        return output

def create_network():
    # Creating pixelCNN architecture
    inputs = T.matrix('features')
    x = inputs / 255.
    #y = T.itensor4('targets')
    conv_list = [ConvolutionalNoFlip(*first_layer, mask='A')]
    for i in range(n_layer):
        conv_list.extend([ConvolutionalNoFlip(*second_layer, mask='B'), Rectifier()])

    conv_list.extend([ConvolutionalNoFlip((3,3), 64, 32, mask='B'), Rectifier()])
    conv_list.extend([ConvolutionalNoFlip((3,3), 64, 64, mask='B'), Rectifier()])
    conv_list.extend([ConvolutionalNoFlip((1,1), 128, 64, mask='B'), Rectifier()])
    conv_list.extend([ConvolutionalNoFlip((1,1), 256, 128, mask='B')])

    sequence = ConvolutionalSequence(
        conv_list,
        num_channels=n_channel,
        batch_size=batch_size,
        image_size=(img_dim,img_dim),
        border_mode='half',
        weights_init=IsotropicGaussian(std=0.05, mean=0),
        biases_init=Constant(0.02),
        tied_biases=False
    )
    sequence.initialize()
    x = sequence.apply(x.reshape((batch_size, n_channel, img_dim, img_dim)))
    x = x.dimshuffle(1,0,2,3)
    x = x.flatten(ndim=3)
    x = x.flatten(ndim=2)
    x = x.dimshuffle(1,0)
    y_hat = Softmax().apply(x)

    cost = CategoricalCrossEntropy().apply(T.cast(inputs.flatten(), 'int64'), y_hat)
    cost.name = 'cross_entropy'
    return cost

def prepare_opti(cost, test):
    model = Model(cost)
    print "Model created"

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
        Checkpoint(path, after_epoch=True)
    ]

    if resume:
        print "Restoring from previous breakpoint"
        extensions.extend([
            Load(path)
        ])

    return model, algorithm, extensions

# Sampler used to sample from the discret distribution of the softmax
class Sampler(Random):

    @application
    def apply(self, featuremap):
        f = self.theano_rng.multinomial(pvals=featuremap, dtype=theano.config.floatX)
        f = T.argmax(f, axis=1) / 255.
        return f.reshape((batch_size, n_channel, img_dim, img_dim))

def sampling(model, input=None, location=(0,0,0)):
    # Sample image from the learnt model
    # model: trained model
    # input: input image to start the reconstruction
    # location: (x, y, channel) tuple for the location of the first pixel to predict
    # x for row, y for columns

    net_output = VariableFilter(roles=[OUTPUT])(model.variables)[-2]
    pred = Sampler(theano_seed=seed).apply(net_output)
    forward = ComputationGraph(pred).get_theano_function()

    # Need to replace by a scan??
    output = np.zeros((batch_size, n_channel, img_dim, img_dim), dtype=np.float32)
    x, y, c = location
    if input is not None:
        output[:,:c+1,:x,:y] = input[:,:c+1,:x,:y]
    for row in range(x, img_dim):
        col_ind = y * (row == x) # Start at column y for the first row
        for col in range(col_ind, img_dim):
            for chan in range(n_channel):
                prediction = forward(output)
                output[:,chan,row,col] = prediction[:,chan,row,col]
    return output

if __name__ == '__main__':
    mnist = MNIST(("train",))
    mnist_test = MNIST(("test",))
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
        dump(main_loop.model, open('pixelcnn.pkl', 'w'))
    else:
        model = load(open('pixelcnn.pkl', 'r'))
