from keras import ops
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.rnn import RNN


class SNNCell(Layer, DropoutRNNCell):
    """ """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_size = None  # set in build

    def build(self, inputs_shape):
        self.state_size = inputs_shape[1:]
        self.built = True

    def call(self, inputs, states, training=False):
        raise NotImplementedError

    def compute_output_shape(self, inputs_shape, states_shape=None):
        return inputs_shape, inputs_shape

    def get_initial_state(self, batch_size=None):
        return [ops.zeros((batch_size, self.state_size), dtype=self.compute_dtype)]


class LIFCell(SNNCell):
    def __init__(self, th=1.0, tau=2.0, surrogate="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.th = th
        self.tau = tau
        self.spike_fn = surrogate  # get surrogate fn

    def call(self, inputs, states, training=False):
        v = v + (x - v) / self.tau
        x = self.spike_fn(v - self.th)
        v = v * (1 - x)
        return x, v

    def get_config(self):
        config = {
            "th": self.th,
            "tau": self.tau,
            "surrogate": self.spike_fn,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class SNN(RNN):
    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        output_size = getattr(self.cell, "output_size", None)
        if output_size is None:
            output_size = self.state_size[0]
        if not isinstance(output_size, int):
            raise ValueError("output_size must be an integer.")
        if self.return_sequences:
            output_shape = (sequences_shape[0], sequences_shape[1], output_size)
        else:
            output_shape = (sequences_shape[0], output_size)
        if self.return_state:
            return output_shape, *state_shape
        return output_shape
