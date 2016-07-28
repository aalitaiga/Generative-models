import numpy as np
from scipy.misc import imsave
import theano
from theano import tensor as T
import random

from blocks.bricks.conv import ConvolutionalSequence, Convolutional
from blocks.bricks import Random, application
from blocks.extensions import SimpleExtension
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.serialization import dump
from blocks.roles import OUTPUT

class SaveModel(SimpleExtension) :
    def __init__(self, name, **kwargs) :
        super(SaveModel, self).__init__(**kwargs)
        self.name = name

    def do(self, which_callback, *args) :
        model = self.main_loop.model
        f = open(self.name + '_epoch_' +
            str(self.main_loop.log.status['epochs_done']) + '.pkl', 'w')
        dump(model, f)
        f.close()

class ApplyMask(SimpleExtension) :
    def __init__(self, *args, **kwargs) :
        super(ApplyMask, self).__init__(*args, **kwargs)

    def do(self, which_callback, *args) :
        # reset part of the kernel to 0 as the PixelCNN paper
        for brick in self.main_loop.model.get_top_bricks() :
            if isinstance(brick, ConvolutionalSequence):
                convseq = brick
                break

        for brick in convseq.children :
            if isinstance(brick, Convolutional):
                brick.W.set_value(brick.W.get_value() * brick.mask)

# Sampler used to sample from the discret distribution of the softmax
class SamplerMultinomial(Random):

    @application
    def apply(self, featuremap):
        from pixelblocks import img_dim, n_channel, batch_size

        f = self.theano_rng.multinomial(pvals=featuremap, dtype=theano.config.floatX)
        f = T.argmax(f, axis=1)
        return f.reshape((batch_size, n_channel, img_dim, img_dim))

class SamplerBinomial(Random):

    @application
    def apply(self, featuremap):
        # featuremap = featuremap.reshape((-1, 784))
        # import ipdb; ipdb.set_trace()
        return self.theano_rng.uniform(size=featuremap.shape,dtype=theano.config.floatX) < featuremap
        # return sampled_output.reshape((-1,1,28,28))

class GenerateSamples(SimpleExtension):
    def __init__(self, *args, **kwargs):
        super(GenerateSamples, self).__init__(*args, **kwargs)

    def do(self, which_callback, *args):
        from gatedpixelblocks import n_channel, batch_size, img_dim, MODE, path, dataset

        model = self.main_loop.model
        net_output = VariableFilter(roles=[OUTPUT])(model.variables)[-2]
        print '{} output used'.format(net_output)
        # import ipdb; ipdb.set_trace()
        Sampler = SamplerMultinomial if MODE == '256ary' else SamplerBinomial
        pred = Sampler(theano_seed=random.randint(0,1000)).apply(net_output)
        forward = ComputationGraph(pred).get_theano_function()

        # Need to replace by a scan??
        output = np.zeros((batch_size, n_channel, img_dim, img_dim), dtype=np.float32)
        x, y, c = (0,0,0)  # location
        # if input_ is not None:
        #     output[:,:c+1,:x,:y] = input_[:,:c+1,:x,:y]
        for row in range(x, img_dim):
            col_ind = y * (row == x)  # Start at column y for the first row to predict
            for col in range(col_ind, img_dim):
                for chan in range(n_channel):
                    prediction = forward(output)[0]
                    output[:,chan,row,col] = prediction[:,chan,row,col]

        output = output.reshape((4, 4, n_channel, img_dim, img_dim)).transpose((1,3,0,4,2))
        if n_channel == 1:
            output = output.reshape((4*img_dim,4*img_dim))
        else:
            output = output.reshape((4*img_dim,4*img_dim,n_channel))
        imsave(
            path+'/'+'{}_samples_epoch{}.jpg'.format(dataset, str(self.main_loop.log.status['epochs_done'])),
            output
        )
