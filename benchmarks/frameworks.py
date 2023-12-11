from .utils import Benchmarkable


class Keras_SNN_JAX(Benchmarkable):
    def __init__(self) -> None:
        name = "Keras_SNN_JAX"
        super().__init__(name)

    def __call__(self, train_data, test_data) -> None:
        import keras
        import keras_snn as snn

        def get_model():
            model = keras.models.Sequential()
            model.add(keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu"))
            model.add(keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu"))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(512))
            model.add(keras.layers.RNN(snn.LIFCell()))
            model.add(keras.layers.Dense(10))


# class Spyx(Benchmarkable):
#     def __init__(self) -> None:
#         name = "Spyx"
#         super().__init__(name)

#     def __call__(self, train_data, test_data):
#         import spyx
#         from spyx import nn as snn
#         import haiku as hk
#         import jax
#         from jax import numpy as jnp
#         import optax

#         def model(x):
#             x = hk.BatchApply(hk.Conv2D(32, kernel_shape=3, stride=2, with_bias=False))(
#                 x
#             )
#             core = hk.DeepRNN(
#                 [
#                     snn.LIF(
#                         (
#                             32,
#                             17,
#                             17,
#                         ),
#                         activation=spyx.axn.Axon(spyx.axn.sigmoid()),
#                     ),
#                     hk.AvgPool(window_shape=17, strides=1, padding="VALID"),
#                     hk.Flatten(2),
#                     snn.LIF((64,), activation=spyx.axn.Axon(spyx.axn.sigmoid())),
#                     hk.Linear(10, with_bias=False),
#                     snn.LI((10,)),
#                 ]
#             )

#             spikes, V = hk.dynamic_unroll(
#                 core, x, core.initial_state(x.shape[0]), time_major=False, unroll=32
#             )

#             return spikes, V

#         key = jax.random.PRNGKey(0)
#         model = hk.without_apply_rng(hk.transform_with_state(model))
#         params, reg_init = model.init(rng=key, x=train_data[0])

#         aug = spyx.data.shift_augment(max_shift=16)

#         opt = optax.chain(
#             optax.centralize(),
#             optax.adam(1e-3),
#         )
#         opt_state = opt.init(params)

#         @self.jax.jit
#         def net_eval(weights, events, targets):
#             readout, spike_counts = model.apply(weights, reg_init, events)
#             traces, V_f = readout
#             return spyx.fn.integral_crossentropy(traces, targets)

#         surrogate_grad = jax.value_and_grad(net_eval)
#         rng = jax.random.PRNGKey(0)

#         # compile the meat of our training loop for speed
#         @jax.jit
#         def train_step(state, data):
#             grad_params, opt_state = state
#             events, targets = data  # fix this
#             events = jnp.unpackbits(events, axis=1)  # decompress temporal axis
#             # compute loss and gradient                    # need better augment rng
#             loss, grads = surrogate_grad(
#                 grad_params,
#                 aug(events, jax.random.fold_in(rng, jnp.sum(targets))),
#                 targets,
#             )
#             # generate updates based on the gradients and optimizer
#             updates, opt_state = opt.update(grads, opt_state, grad_params)
#             # return the updated parameters
#             new_state = [optax.apply_updates(grad_params, updates), opt_state]
#             return new_state, loss

#         # For validation epochs, do the same as before but compute the
#         # accuracy, predictions and losses (no gradients needed)
#         @jax.jit
#         def eval_step(grad_params, data):
#             events, targets = data # fix
#             events = jnp.unpackbits(events, axis=1)
#             readout, spike_counts = model.apply(grad_params, reg_init, events)
#             traces, V_f = readout
#             acc, pred = spyx.fn.integral_accuracy(traces, targets)
#             loss = spyx.fn.integral_crossentropy(traces, targets)
#             return grad_params, jnp.array([acc, loss])
