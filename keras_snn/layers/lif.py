import keras
from keras.layers import Layer

from keras_snn import backend


class LIF(Layer):
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

    def __init__(self, th=1.0, tau=2.0, surrogate="sigmoid", **kwargs):
        super().__init__()
        self.v = 0.0
        self.th = th
        self.tau = tau
        self.spike_fn = backend.activations.spike_fn(surrogate, **kwargs)

    def __call__(self, x):
        x = keras.ops.convert_to_tensor(x)
        self.v = self.v + (x - self.v) / self.tau
        x = self.spike_fn(self.v - self.th)
        self.v = self.v * (1 - x)
        return x

    def reset_states(self):
        # Compatibility alias.
        self.reset_state()

    def reset_state(self):
        if self.states is not None:
            for v in self.states:
                v.assign(keras.ops.zeros_like(v))
