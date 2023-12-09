from keras.src.backend.config import backend

if backend() == "torch":
    # When using the torch backend,
    # torch needs to be imported first, otherwise it will segfault
    # upon import.
    import torch

elif backend() == "jax":
    from keras_snn.backend.jax import *
elif backend() == "torch":
    from keras_snn.backend.torch import *

else:
    raise ValueError(f"Unable to import backend : {backend()}")