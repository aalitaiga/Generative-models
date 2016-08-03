""" Implementing pixelCNN in Blocks"""
import argparse
from datetime import date
import logging
import os
import sys

from blocks.algorithms import GradientDescent, Adam, RMSProp, AdaGrad
from blocks.bricks.conv import ConvolutionalSequence, Convolutional
from blocks.bricks import application, Logistic, Rectifier, Softmax, Initializable
from blocks.bricks.cost import BinaryCrossEntropy, CategoricalCrossEntropy
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Load
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.serialization import dump, load

from fuel.datasets import BinarizedMNIST, MNIST, CIFAR10
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
import numpy as np
import theano
from theano import tensor as T


from utils import SaveModel, GenerateSamples

sys.setrecursionlimit(500000)

batch_size = 16
dataset = "binarized_mnist"
if dataset in ("mnist", "binarized_mnist"):
    img_dim = 28
    n_channel = 1
elif dataset == "cifar10":
    img_dim = 32
    n_channel = 3
MODE = "binary" if dataset == "binarized_mnist" else "256ary"
path = 'pixelcnn_{}_{}'.format(dataset, date.today())

if not os.path.exists(path):
    os.makedirs(path)

logging.basicConfig(filename=path+'/'+path+'.log',
    level=logging.INFO,
    format='%(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

logger = logging.getLogger(__name__)

nb_epoch = 450
patience = 5
check = path+'/'+'checkpoint_{}.pkl'.format(dataset)
sources = ('features',)
train = True
resume = False
save_every = 10  # Save model every m-th epoch
gen_every = 1
seed = 2

n_layer = 5
h = 32
first_layer = ((7, 7), h*n_channel)
second_layer = ((3, 3), h*n_channel)
third_layer = ((1, 1), 256*n_channel) if MODE == '256ary' else ((1, 1), n_channel)


class ConvolutionalNoFlip(Convolutional):
    def __init__(self, *args, **kwargs):
        self.mask_type = kwargs.pop('mask_type', None)
        Convolutional.__init__(self, *args, **kwargs)

    def push_allocation_config(self):
        super(ConvolutionalNoFlip, self).push_allocation_config()
        if self.mask_type:
            assert self.filter_size[0] == 1
            filter_shape = (self.num_filters, self.num_channels) + self.filter_size
            mask = np.ones(filter_shape, dtype=theano.config.floatX)
            for i in xrange(n_channel):
                for j in xrange(n_channel):
                    if (self.mask_type == 'A' and i >= j) or (self.mask_type == 'B' and i > j):
                        mask[
                            j::n_channel,
                            i::n_channel,
                            0,
                            -1
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

        if self.mask_type:
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

class GatedPixelCNN(Initializable):
    def __init__(self, name, filter_size, num_channels, num_filters=None, batch_size=None,
                 res=True, image_size=(None, None), tied_biases=None, **kwargs):
        # TODO: Activation in 1x1??
        super(GatedPixelCNN, self).__init__(**kwargs)
        if num_filters is None:
            num_filters = num_channels
        self.name = name
        self.image_size = image_size
        self.tied_biases = tied_biases
        self.res = res
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.vertical_conv_nxn = ConvolutionalNoFlip(
            filter_size=((filter_size//2)+1,filter_size),
            num_filters=2*num_filters,
            num_channels=num_channels,
            border_mode=(filter_size // 2 + 1, filter_size // 2),
            name=name+"v_nxn"
        )
        self.vertical_conv_1x1 = ConvolutionalNoFlip(
            filter_size=(1,1),
            num_filters=2*num_filters,
            num_channels=2*num_filters,
            border_mode='valid',
            name=name+"v_1x1"
        )
        self.horizontal_conv_1xn = ConvolutionalNoFlip(
            filter_size=(1,(filter_size//2)+1),
            num_filters=2*num_filters,
            num_channels=num_channels,
            border_mode=(0,filter_size//2),
            mask_type='B' if self.res else 'A',
            name=name+"h_1xn"
        )
        self.children = [self.vertical_conv_nxn, self.vertical_conv_1x1,
            self.horizontal_conv_1xn]
        if self.res or True:
            self.horizontal_conv_1x1 = ConvolutionalNoFlip(
                filter_size=(1,1),
                num_filters=num_filters,
                num_channels=num_filters,
                name=name+"h_1x1",
                batch_size=batch_size,
                mask_type='B',
            )
            self.children.append(self.horizontal_conv_1x1)

    def push_allocation_config(self):
        for child in self.children:
            child.image_size = self.image_size
            child.batch_size = self.batch_size
            child.tied_biases = self.tied_biases
        super(GatedPixelCNN, self).push_allocation_config()


    @application(inputs=['input_v', 'input_h'], outputs=['output_v', 'output_h'])
    def apply(self, input_v, input_h):
        # Vertical stack
        v_nxn_out = self.vertical_conv_nxn.apply(input_v)
        # Different cropping are used depending on the row we wish to condition on
        v_nxn_out_to_h = v_nxn_out[:,:,:-(self.filter_size//2)-2,:]
        v_nxn_out_to_v = v_nxn_out[:,:,1:-(self.filter_size//2)-1,:]
        v_1x1_out = self.vertical_conv_1x1.apply(v_nxn_out_to_h)
        output_v = T.tanh(v_nxn_out_to_v[:,:self.num_filters,:,:]) * \
            T.nnet.sigmoid(v_nxn_out_to_v[:,self.num_filters:,:,:])

        # Horizontal stack
        h_1xn_out = self.horizontal_conv_1xn.apply(input_h)
        h_1xn_out = h_1xn_out[:,:,:,:-(self.filter_size//2)]
        h_sum = h_1xn_out + v_1x1_out
        h_activation = T.tanh(h_sum[:,:self.num_filters,:,:]) * \
            T.nnet.sigmoid(h_sum[:,self.num_filters:,:,:])
        h_1x1_out = self.horizontal_conv_1x1.apply(h_activation)
        if self.res:
            # input_h_padded = T.zeros(input_h.shape, dtype=theano.config.floatX)
            # input_h_padded = T.inc_subtensor(input_h_padded[:,:,3:,3:], input_h[:,:,:-3,:-3])
            # input_h = input_h_padded
            output_h = h_1x1_out #+ input_h
        else:
            output_h = h_1x1_out #h_activation
        return output_v, output_h


def create_network(inputs=None, batch=batch_size):
    if inputs is None:
        inputs = T.tensor4('features')
    x = T.cast(inputs,'float32')
    x = x / 255. if dataset != 'binarized_mnist' else x

    # GatedPixelCNN
    gated = GatedPixelCNN(
        name='gated_layer_0',
        filter_size=7,
        image_size=(img_dim,img_dim),
        num_filters=h*n_channel,
        num_channels=n_channel,
        batch_size=batch,
        weights_init=IsotropicGaussian(std=0.02, mean=0),
        biases_init=Constant(0.02),
        res=False
    )
    gated.initialize()
    x_v, x_h = gated.apply(x, x)

    for i in range(n_layer):
        gated = GatedPixelCNN(
            name='gated_layer_{}'.format(i+1),
            filter_size=3,
            image_size=(img_dim,img_dim),
            num_channels=h*n_channel,
            batch_size=batch,
            weights_init=IsotropicGaussian(std=0.02, mean=0),
            biases_init=Constant(0.02),
            res=True
        )
        gated.initialize()
        x_v, x_h = gated.apply(x_v, x_h)

    conv_list = []
    conv_list.extend([Rectifier(), ConvolutionalNoFlip((1,1), h*n_channel, mask_type='B', name='1x1_conv_1')])
    #conv_list.extend([Rectifier(), ConvolutionalNoFlip((1,1), h*n_channel, mask='B', name='1x1_conv_2')])
    conv_list.extend([Rectifier(), ConvolutionalNoFlip(*third_layer, mask_type='B', name='output_layer')])

    sequence = ConvolutionalSequence(
        conv_list,
        num_channels=h*n_channel,
        batch_size=batch,
        image_size=(img_dim,img_dim),
        border_mode='half',
        weights_init=IsotropicGaussian(std=0.02, mean=0),
        biases_init=Constant(0.02),
        tied_biases=False
    )
    sequence.initialize()
    x = sequence.apply(x_h)
    if MODE == '256ary':
        x = x.reshape((-1, 256, n_channel, img_dim, img_dim)).dimshuffle(0,2,3,4,1)
        x = x.reshape((-1, 256))
        x_hat = Softmax().apply(x)
        inp = T.cast(inputs, 'int64').flatten()
        cost = CategoricalCrossEntropy().apply(inp, x_hat) * img_dim * img_dim
        cost_bits_dim = categorical_crossentropy(log_softmax(x), inp)
    else:
        x_hat = Logistic().apply(x)
        cost = BinaryCrossEntropy().apply(inputs, x_hat) * img_dim * img_dim
        #cost = T.nnet.binary_crossentropy(x_hat, inputs)
        #cost = cost.sum() / inputs.shape[0]
        cost_bits_dim = -(inputs * T.log2(x_hat) + (1.0 - inputs) * T.log2(1.0 - x_hat)).mean()

    cost_bits_dim.name = "nnl_bits_dim"
    cost.name = 'loglikelihood_nat'
    return cost, cost_bits_dim

# Log of the softmax
def log_softmax(x):
    xdev = x - x.max(axis=1)[:, None]
    lsm = xdev - T.log2(T.sum(T.exp(xdev), axis=1, keepdims=True))
    return lsm

# Categorical cross entropy for log_softmax inputs
def categorical_crossentropy(pred, inputs):
    loss_bits_dim = - T.mean(pred[T.arange(inputs.shape[0]), inputs])
    return loss_bits_dim

def prepare_opti(cost, test, *args):
    model = Model(cost)
    logger.info("Model created")

    algorithm = GradientDescent(
        cost=cost,
        parameters=model.parameters,
        step_rule=Adam(learning_rate=0.015),
        on_unused_sources='ignore'
    )

    to_monitor = [algorithm.cost]
    if args:
        to_monitor.extend(args)

    extensions = [
        FinishAfter(after_n_epochs=nb_epoch),
        FinishIfNoImprovementAfter(notification_name='loglikelihood_nat', epochs=patience),
        TrainingDataMonitoring(
            to_monitor,
            prefix="train",
            after_epoch=True),
        DataStreamMonitoring(
            to_monitor,
            test_stream,
            prefix="test"),
        Printing(),
        ProgressBar(),
        SaveModel(name=path+'/'+'pixelcnn_{}'.format(dataset), every_n_epochs=save_every),
        GenerateSamples(every_n_epochs=gen_every),
        #Checkpoint(path+'/'+'exp.log', save_separately=['log'], every_n_epochs=save_every),
    ]

    if resume:
        logger.info("Restoring from previous checkpoint")
        extensions = [Load(path+'/'+check)]

    return model, algorithm, extensions

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
        data = MNIST(("train",), sources=('features',))
        data_test = MNIST(("test",), sources=('features',))
    elif dataset == 'binarized_mnist':
        data = BinarizedMNIST(("train",), sources=('features',))
        data_test = BinarizedMNIST(("test",), sources=('features',))
    elif dataset == "cifar10":
        data = CIFAR10(("train",))
        data_test = CIFAR10(("test",))

    training_stream = DataStream(
        data,
        iteration_scheme=ShuffledScheme(data.num_examples, batch_size)
    )
    test_stream = DataStream(
        data_test,
        iteration_scheme=ShuffledScheme(data_test.num_examples, batch_size)
    )
    logger.info("Dataset: {} loaded".format(dataset))

    if train:
        cost, cost_bits_dim = create_network()
        model, algorithm, extensions = prepare_opti(cost, test_stream, cost_bits_dim)
        # import ipdb; ipdb.set_trace()
        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=training_stream,
            model=model,
            extensions=extensions
        )
        main_loop.run()
        with open(path+'/'+'pixelcnn.pkl', 'w') as f:
            dump(main_loop.model, f)
        model = main_loop.model
    else:
        model = load(open('pixelcnn_cifar10_2016-07-19/pixelcnn_cifar10_epoch_165.pkl', 'r'))
