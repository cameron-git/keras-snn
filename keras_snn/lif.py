import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import jax
from jax import numpy as jnp


def forward(ctx, x, v, th, tau):
    v = v + (x - v) / tau
    x = (v >= th).to(x)
    ctx.save_for_backward(x, v, th, tau)
    v = v * (1 - x)
    return x, v


@jax.custom_vjp
def sg(x):
    return jnp.heaviside(x, 0)


def sg_fwd(x):
    return sg(x), x


def sg_bwd(res, g):
    (x,) = res
    return g * jax.nn.sigmoid(x) * (1 - jax.nn.sigmoid(x))


sg.defvjp(sg_fwd, sg_bwd)


class LIF(keras.layers.Layer):
    def __init__(self, th=1.0, tau=2.0):
        super().__init__()
        self.v = 0.0
        self.th = th
        self.tau = tau

    def call(self, x):
        self.v = self.v + (x - self.v) / self.tau
        x = sg(self.v - self.th)
        self.v = self.v * (1 - x)
        return x
