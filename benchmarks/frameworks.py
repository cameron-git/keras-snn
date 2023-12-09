from .utils import Benchmarkable

class Keras_SNN_JAX(Benchmarkable):
    def __init__(self) -> None:
        name = "Keras_SNN_JAX"
        super().__init__(name)

    def __call__(self, data) -> None:
        pass