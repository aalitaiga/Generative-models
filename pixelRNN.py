""" Implementation of Pixel Recurrent Neural Networks
http://arxiv.org/pdf/1601.06759v2.pdf
"""

from theano import tensor

from blocks.bricks import application, lazy, Initializable, Logistic, Tanh
from blocks.bricks.recurrent import BaseRecurrent, Bidirectional, recurrent
from blocks.bricks.conv import Convolutional
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.role import add_role, WEIGHT, INITIAL_STATE

from fuel.datasets import MNIST


class LSTMConv(BaseRecurrent, Initializable):

    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None, **kwargs):
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = [self.activation, self.gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(LSTMConv, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name == 'states':
            return self.dim
        if name == 'mask':
            return 0
        return super(LSTMConv, self).get_dim(name)

    def _allocate(self):
        self.W_ss = shared_floatx_nans((self.dim, 4*self.dim), name='W_ss')
        self.W_is = shared_floatx_nans((self.dim,), name='W_is')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        add_role(self.W_ss, WEIGHT)
        add_role(self.W_is, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)

        self.parameters = [
            self.W_ss, self.W_is, self.initial_state_]

    def _initialize(self):
        for weights in self.parameters[:2]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               contexts=[], outputs=['states'])
    def apply(self, inputs, states, mask=None):
        """Apply the Long Short Term Memory transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features * 4). The `inputs` needs to be four times the
            dimension of the LSTM brick to insure each four gates receive
            different transformations of the input. See [Grav13]_
            equations 7 to 10 for more details. The `inputs` are then split
            in this order: Input gates, forget gates, cells and output
            gates.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        .. [Grav13] Graves, Alex, *Generating sequences with recurrent*
            *neural networks*, arXiv preprint arXiv:1308.0850 (2013).
        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.
        """
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        activation = tensor.dot(states, self.W_ss) + tensor.dot(inputs, self.W_is)
        out_gate = self.gate_activation.apply(slice_last(activation, 0))
        forget_gate = self.gate_activation.apply(slice_last(activation, 1))
        in_gate = self.gate_activation.apply(slice_last(activation, 2))

        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(
            slice_last(activation, 3) + next_cells * self.W_cell_to_out)
        next_states = out_gate * self.activation.apply(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]

class BidirectionalLSTM(Bidirectional):
    """ Wrap two convolutional LSTM"""

    @application
    def apply(self):
        pass
