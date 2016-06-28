from blocks.extensions import SimpleExtension
from blocks.serialization import dump

class SaveModel(SimpleExtension) :
    def __init__(self, name, **kwargs) :
        super(SaveModel, self).__init__(**kwargs)
        self.name = name

    def do(self, which_callback, *args) :
        model = self.main_loop.model
        dump(model, open(self.name + '_epoch_' +
            str(self.main_loop.log.status['epochs_done']) + '.pkl', 'w'))
