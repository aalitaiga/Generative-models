""" Implementing pixelCNN in Blocks"""
import argparse
import sys

from blocks.algorithms import GradientDescent, Adam, RMSProp, AdaGrad
from blocks.bricks.conv import ConvolutionalSequence, Convolutional
from blocks.bricks import application, Logistic, Rectifier, Softmax, Random
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy
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
from fuel.datasets import BinarizedMNIST, MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
import numpy as np
import theano
from theano import tensor as T
from scipy.misc import imsave

from utils import SaveModel

sys.setrecursionlimit(500000)

batch_size = 16
dataset = "mnist"
img_dim = 28
nb_epoch = 25
n_channel = 1
patience = 3
path = 'checkpoint.pkl'
sources = ('features',)
train = True
resume = False
save_every = 5  # Save model every m-th epoch
seed = 2

MODE = '256ary'  # choice with 'binary' and '256ary

n_layer = 3
res_connections = False
h = 32
first_layer = ((7, 7), h*n_channel, n_channel)
second_layer = ((3, 3), h*n_channel, h*n_channel)
if MODE == '256ary':
    third_layer = ((1, 1), 256*n_channel, h*n_channel)
else:
    third_layer = ((1, 1), 1, h*n_channel)

class ConvolutionalNoFlip(Convolutional) :
    def __init__(self, *args, **kwargs):
        self.mask_type = kwargs.pop('mask', None)
        Convolutional.__init__(self, *args, **kwargs)

        if self.mask_type:
            filter_shape = (self.num_filters, self.num_channels) + self.filter_size
            mask = np.ones(filter_shape, dtype=theano.config.floatX)
            center = filter_shape[2] // 2

            # Channels are split to have access to different information from the past
            mask[:,:,center+1:,:] = 0.
            mask[:,:,center,center+1:] = 0.
            for i in xrange(self.num_channels):
                for j in xrange(self.num_channels):
                    if (self.mask == 'A' and i >= j) or (self.mask == 'B' and i > j):
                        mask[
                            j::self.num_channels,i::self.num_channels,center,center
                        ] = 0.
            self.mask = mask

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

        self.W.set_value(self.W.get_value() * self.mask)
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

class ConvolutionalNoFlipWithRes(ConvolutionalNoFlip):

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        output = ConvolutionalNoFlip.apply(self, input_)
        return input_ + output if res_connections else output

def create_network(inputs=None, batch=batch_size):
    if inputs is None:
        inputs = T.tensor4('features')
    x = T.cast(inputs,'float32') / 255. if dataset != 'binarized_mnist' else inputs

    # PixelCNN architecture
    conv_list = [ConvolutionalNoFlip(*first_layer, mask='A', name='0'), Rectifier()]
    for i in range(n_layer):
        conv_list.extend([ConvolutionalNoFlipWithRes(*second_layer, mask='B', name=str(i+1)), Rectifier()])

    conv_list.extend([ConvolutionalNoFlip((1,1), h*n_channel, h*n_channel, mask='B', name=str(n_layer+1)), Rectifier()])
    conv_list.extend([ConvolutionalNoFlip((1,1), 128*n_channel, h*n_channel, mask='B', name=str(n_layer+2)), Rectifier()])
    conv_list.extend([ConvolutionalNoFlip((1,1), 256*n_channel, 128*n_channel, mask='B', name=str(n_layer+3))])

    sequence = ConvolutionalSequence(
        conv_list,
        num_channels=n_channel,
        batch_size=batch,
        image_size=(img_dim,img_dim),
        border_mode='half',
        weights_init=IsotropicGaussian(std=0.05, mean=0),
        biases_init=Constant(0.02),
        tied_biases=False
    )
    sequence.push_initialization_config()
    sequence.initialize()
    x = sequence.apply(x)
    if MODE == '256ary':
        #x = x.reshape((-1, 256, n_channel, img_dim, img_dim)).dimshuffle(0,2,3,4,1).reshape((-1,256))
        x = x.dimshuffle(1,0,2,3)
        x = x.flatten(ndim=3)
        x = x.flatten(ndim=2)
        x = x.dimshuffle(1,0)
        x_hat = Softmax().apply(x)
        cost = CategoricalCrossEntropy().apply(T.cast(inputs.flatten(), 'int64'), x_hat)
    else:
        x_hat = Logistic().apply(x)
        cost = BinaryCrossEntropy().apply(inputs, x_hat)
    cost.name = 'pixelcnn_cost'
    return cost

# Sampler used to sample from the discret distribution of the softmax
class SamplerMultinomial(Random):

    @application
    def apply(self, featuremap, batch=batch_size):
        f = self.theano_rng.multinomial(pvals=featuremap, dtype=theano.config.floatX)
        f = T.argmax(f, axis=1)
        return f.reshape((batch, n_channel, img_dim, img_dim))

class SamplerBinomial(Random):

    @application
    def apply(self, featuremap, batch=batch_size):
        return T.lt(self.theano_rng.uniform(size=(batch, 1, img_dim, img_dim)), featuremap)

def sampling(model, input_=None, location=(0,0,0), batch=batch_size):
    # Sample image from the learnt model
    # model: trained model
    # input: input image to start the reconstruction
    # location: (x, y, channel) tuple for the location of the first pixel to predict
    # x for row, y for columns

    net_output = VariableFilter(roles=[OUTPUT])(model.variables)[-2]
    print 'Output used: {}'.format(net_output)
    Sampler = SamplerMultinomial if MODE == '256ary' else SamplerBinomial
    pred = Sampler(theano_seed=seed).apply(net_output, batch=batch)
    forward = ComputationGraph(pred).get_theano_function()

    # Need to replace by a scan??
    output = np.zeros((batch, n_channel, img_dim, img_dim), dtype=np.float32)
    x, y, c = location
    if input_ is not None:
        output[:,:c+1,:x,:y] = input_[:,:c+1,:x,:y]
    for row in range(x, img_dim):
        col_ind = y * (row == x)  # Start at column y for the first row to predict
        for col in range(col_ind, img_dim):
            for chan in range(n_channel):
                prediction = forward(output)[0]
                output[:,chan,row,col] = prediction[:,chan,row,col]
    return output

def prepare_opti(cost, test):
    model_ = Model(cost)
    print "Model created"

    algorithm = GradientDescent(
        cost=cost,
        parameters=model_.parameters,
        step_rule=Adam(),
        on_unused_sources='ignore'
    )

    extensions = [
        FinishAfter(after_n_epochs=nb_epoch),
        FinishIfNoImprovementAfter(notification_name='test_pixelcnn_cost', epochs=patience),
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
        Checkpoint(path, every_n_epochs=save_every),
        SaveModel(name='pixelcnn', every_n_epochs=save_every)
    ]

    if resume:
        print "Restoring from previous breakpoint"
        extensions.extend([
            Load(path)
        ])

    return model_, algorithm, extensions

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # action = parser.add_mutually_exclusive_group()
    # action.add_argument('-t', '--train', help="Start training the model")
    # action.add_argument('-s', '--sample', help='Sample images from the trained model')
    #
    # parser.add_argument('--experiment', nargs=1, type=str,
    #     help="Change default location to run experiment")
    # parser.add_argument('--path', nargs=1, type=str,
    #     help="Change default location to save model")

    if dataset == 'mnist':
        data = MNIST(("train",))
        data_test = MNIST(("test",))
    elif dataset == 'binarized_mnist':
        data = BinarizedMNIST(("train",))
        data_test = BinarizedMNIST(("test",))
    else:
        pass  # Add CIFAR 10
    training_stream = DataStream(
        data,
        iteration_scheme=ShuffledScheme(data.num_examples, batch_size)
    )
    test_stream = DataStream(
        data_test,
        iteration_scheme=ShuffledScheme(data_test.num_examples, batch_size)
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
        with open('pixelcnn.pkl', 'w') as f:
            dump(main_loop.model, f)
        model = main_loop.model
    else:
        model = load(open('pixelcnn_epoch_5.pkl', 'r'))

    # Generate some samples
    samples = sampling(model)
    samples = samples.reshape((16*28,28))

    imsave('{}_samples.jpg'.format(dataset), samples)
