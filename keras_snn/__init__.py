import os
import warnings

supported_backends = ["jax", "torch"]

if os.environ.get("KERAS_BACKEND") not in supported_backends:
    warnings.warn("Keras backend not set. Defaulting to JAX.")
    os.environ["KERAS_BACKEND"] = "jax"

from . import backend
from . import layers
