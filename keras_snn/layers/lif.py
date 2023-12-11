import keras
from keras.layers import Layer

from keras_snn import backend

# class SNNCell(Layer):
#     def __init__(self, layer, activation, **kwargs):
#         super().__init__(**kwargs)
#         self.layer = layer
#         self.activation = activation

#         def build(self, input_shape):
#             self.layer.build(input_shape)
#             self.activation.build(input_shape)
#             self.built = True

#         def call(self, input_at_t, states_at_t):
#             output_at_t = self.layer(input_at_t)
#             output_at_t = self.activation(output_at_t)
#             return output_at_t, states_at_t


class LIFCell(Layer):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    Args:
        th (float): Threshold value for firing.
        tau (float): Time constant for membrane potential decay.
        surrogate (str): Surrogate function for differentiability.
        **kwargs: Additional keyword arguments for the surrogate function.

    Attributes:
        v (float): Membrane potential of the neuron.

    Returns:
        x (float): Output of the LIF neuron after applying the activation function.
    """

    def __init__(self,  th=1.0, tau=2.0, surrogate="sigmoid", **kwargs):
        super().__init__()
        self.state_size = -1
        self.th = th
        self.tau = tau
        self.spike_fn = backend.activations.spike_fn(surrogate, **kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, x, v):
        v_shape = v[0].shape
        v = v[0].reshape(x.shape)
        v = v + (x - v) / self.tau
        x = self.spike_fn(v - self.th)
        v = v * (1 - x)
        return x, [v.reshape(v_shape)]
