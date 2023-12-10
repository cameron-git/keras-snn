import jax
from jax import numpy as jnp


def spike_fn(surrogate="sigmoid", **kwargs):
    """
    Returns a spiking activation function (Heaviside) with surrogate gradient based on the given surrogate.

    Args:
        surrogate (str): The surrogate function to use. Defaults to "sigmoid".

    Returns:
        function: The spike function.

    Raises:
        ValueError: If the given surrogate is unknown.
    """

    @jax.custom_vjp
    def sg(x):
        return jnp.heaviside(x, 0)

    def sg_fwd(x):
        return sg(x), x

    if surrogate == "sigmoid":
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]

        def sg_bwd(res, g):
            (x,) = res
            x = jax.nn.sigmoid(x)
            return g * x * (1 - x) * alpha

    else:
        raise ValueError(f"Unknown surrogate: {surrogate}")

    sg.defvjp(sg_fwd, sg_bwd)

    return sg
