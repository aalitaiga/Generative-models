from blocks.bricks.conv import ConvolutionalSequence, Convolutional
from blocks.extensions import SimpleExtension
from blocks.serialization import dump

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
